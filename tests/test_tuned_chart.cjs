const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { test } = require('node:test');
const path = require('node:path');

const source = fs.readFileSync(path.join(__dirname, '../src/tabbench_bio/web/js/app.js'), 'utf8')
  .replace('main().catch(showLoadError);', '');

function plot(rows) {
  const models = Object.fromEntries(rows.map(row => [row.model_id, {
    ...row, tuned_from: 'tuned_from' in row ? row.tuned_from
      : row.model_id.endsWith('-TUNED') ? row.model_id.slice(0, -6) : null,
    regular_max_features: null, training_data_overlap: row.training_data_overlap === true,
  }]));
  const context = vm.createContext({ document: { body: { classList: { contains: () => false } } } });
  vm.runInContext(source, context);
  context.input = { models };
  context.rows = rows;
  return vm.runInContext('DATA = input; eloPlotSpec(rows, false, 10000)', context);
}

function row(id, elo) {
  return { model_id: id, display: id, Elo: elo, Elo_lo: elo - 20, Elo_hi: elo + 20,
    category: 'Tree-based', color: '#059669', n_targets: 43 };
}

test('tuned bars end at the tuned rating, including decreases and zero changes', () => {
  for (const tuned of [1200, 800, 1000, -100]) {
    const spec = plot([row('RF', 1000), row('RF-TUNED', tuned), row('XT', 900)]);
    assert.equal(spec.peers.length, 2);
    const [, delta, endpoint] = spec.traces;
    assert.equal(delta.base[0], 1000);
    assert.equal(delta.base[0] + delta.x[0], tuned);
    assert.equal(endpoint.x[0], tuned);
    assert.equal(endpoint.error_x.array[0], tuned < 0 ? null : 20);
    assert.equal(spec.layout.shapes[0].x0, 1000);
    assert.ok(spec.layout.xaxis.range[0] <= (tuned < 0 ? tuned : tuned - 20));
  }
});

test('missing parents remain standalone and input rows are unchanged', () => {
  const rows = [row('RF', 1000), row('XT-TUNED', 1100), row('AUTOGLUON', 1300)];
  const before = JSON.stringify(rows);
  const spec = plot(rows);
  assert.equal(spec.traces.length, 1);
  assert.equal(spec.peers.length, 2);
  assert.equal(spec.autogluon.Elo, 1300);
  assert.equal(JSON.stringify(rows), before);
});

test('pairs use their maximum Elo and a negative parent remains visible', () => {
  const spec = plot([row('RF', 1000), row('RF-TUNED', 800), row('XT', -50), row('XT-TUNED', 1100)]);
  assert.equal(spec.peers[0].model_id, 'RF');
  assert.equal(spec.peers[1].model_id, 'XT');
  assert.equal(spec.hiddenNegativeCount, 0);
  assert.equal(spec.layout.yaxis.categoryarray.length, 2);
});

test('registry parent and overlap metadata work for an arbitrary variant name', () => {
  const variant = { ...row('CUSTOM-SEARCH', 1100), tuned_from: 'RF', training_data_overlap: true };
  const spec = plot([row('RF', 1000), variant]);
  assert.equal(spec.paired.length, 1);
  assert.equal(spec.traces[1].base[0], 1000);
  assert.match(spec.traces[1].customdata[0][0], /†/);
  assert.match(spec.traces[1].customdata[0][6], /training data/);
});


test('paired parents have no error bars or interval tooltip; standalone intervals remain', () => {
  const spec = plot([row('RF', 1000), row('RF-TUNED', 1150), row('XT', 900)]);
  const base = spec.traces[0];
  const parent = spec.peers.findIndex(item => item.model_id === 'RF');
  const standalone = spec.peers.findIndex(item => item.model_id === 'XT');
  assert.equal(base.error_x.array[parent], null);
  assert.equal(base.error_x.arrayminus[parent], null);
  assert.doesNotMatch(base.hovertemplate[parent], /95% interval/);
  assert.equal(base.error_x.array[standalone], 20);
  assert.match(base.hovertemplate[standalone], /95% interval/);
  assert.equal(spec.traces[1].hoverinfo, 'skip');
  assert.doesNotMatch(spec.traces[2].hovertemplate, /Tuning change/);
  assert.equal(spec.traces[2].error_x.array[0], 20);
  assert.equal(spec.layout.showlegend, false);
});

function runtime(rows, elements = {}) {
  const models = Object.fromEntries(rows.map(item => [item.model_id, {
    ...item, tuned_from: item.tuned_from || null, training_data_overlap: false,
  }]));
  const context = vm.createContext({ document: {
    body: { classList: { contains: () => false } },
    getElementById: id => elements[id],
  } });
  vm.runInContext(source, context);
  context.input = { models };
  context.rows = rows;
  vm.runInContext('DATA = input;', context);
  return context;
}

test('line views prefer registry-linked tuned versions and retain standalone models', () => {
  const rows = [row('RF', 1000), { ...row('SEARCH', 1100), tuned_from: 'RF' }, row('XT', 1050)];
  const context = runtime(rows);
  const preferred = vm.runInContext('preferTunedRows(rows)', context);
  assert.equal(JSON.stringify(preferred.map(item => item.model_id)), JSON.stringify(['SEARCH', 'XT']));
  assert.equal(rows.length, 3);
});

test('tuning keys share the bottom family legend only when pairs exist', () => {
  const elements = { 'family-legend': {}, 'elo-overlap-warning': {} };
  const context = runtime([row('RF', 1000)], elements);
  vm.runInContext('renderFamilyLegend(rows, true)', context);
  assert.match(elements['family-legend'].innerHTML, /legend-tuning-change/);
  assert.match(elements['family-legend'].innerHTML, /legend-tuned-elo/);
  vm.runInContext('renderFamilyLegend(rows, false)', context);
  assert.doesNotMatch(elements['family-legend'].innerHTML, /legend-tuning-change/);
});

test('dense timing plots retain every label without overlaps on desktop and mobile', () => {
  const rows = Array.from({ length: 24 }, (_, i) => ({
    ...row('MODEL-' + i, 1000), seconds: 10, f1_macro: 0.8,
  }));
  const context = runtime(rows);
  for (const mobile of [false, true]) {
    const spec = vm.runInContext('costPlotSpec(rows, "f1_macro", { column: "seconds", axis: "Seconds" }, ' + mobile + ')', context);
    assert.equal(spec.traces.slice(1).reduce((n, trace) => n + trace.x.length, 0), rows.length);
    assert.equal(spec.layout.annotations.length, rows.length);
    assert.equal(new Set(spec.layout.annotations.map(a => a.text)).size, rows.length);
    const boxes = spec.layout.annotations.map(a => {
      const w = a.text.length * a.font.size * 0.65 + 10, h = a.font.size + 8;
      return [a.ax - w / 2, a.ay - h / 2, a.ax + w / 2, a.ay + h / 2];
    });
    boxes.forEach((a, i) => boxes.slice(i + 1).forEach(b => {
      assert.ok(!(a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1]), 'labels overlap');
    }));
    assert.ok(spec.traces.slice(1).every(trace => trace.hovertemplate.includes('%{customdata}')));
  }
});


test('displayed Elo intervals stop at zero and preserve original hover bounds', () => {
  const tuned = { ...row('RF-TUNED', 10), Elo_lo: -50, Elo_hi: 40 };
  const standalone = { ...row('XT', 5), Elo_lo: -30, Elo_hi: 20 };
  const spec = plot([row('RF', 1000), tuned, standalone]);
  assert.equal(spec.traces[2].error_x.arrayminus[0], 10);
  const xtIndex = spec.peers.findIndex(item => item.model_id === 'XT');
  assert.equal(spec.traces[0].error_x.arrayminus[xtIndex], 5);
  assert.equal(spec.traces[0].customdata[xtIndex][0], -30);
  assert.equal(spec.traces[2].customdata[0][4], -50);
});

test('negative estimates retain only a nonnegative visible interval portion', () => {
  const crossing = { ...row('RF-TUNED', -10), Elo_lo: -50, Elo_hi: 40 };
  const spec = plot([row('RF', 1000), crossing]);
  assert.equal(spec.traces[2].x[0], -10);
  assert.equal(spec.traces[2].error_x.array[0], null);
  const clipped = spec.traces[3];
  assert.equal(clipped.x[0], 0);
  assert.equal(clipped.error_x.array[0], 40);
  assert.equal(clipped.error_x.arrayminus[0], 0);
  const negative = plot([row('RF', 1000), row('RF-TUNED', -100)]);
  assert.equal(negative.traces.length, 3);
  assert.equal(negative.traces[2].error_x.array[0], null);
});

test('budget curves clip displayed intervals without changing line values', () => {
  const elements = {
    'perf-domain': { value: 'all' }, 'perf-cap': { value: '10000' },
    'perf-group': { value: 'all' }, 'perf-ci': { checked: true },
    'performance-chart': {}, 'performance-note': {},
  };
  const rows = [
    { ...row('RF', 1000), cell: 'one', domain: 'all', metric: 'f1_macro' },
    { ...row('SEARCH', -10), tuned_from: 'RF', Elo_lo: -50, Elo_hi: 40, cell: 'one', domain: 'all', metric: 'f1_macro' },
  ];
  const context = runtime(rows, elements);
  context.window = { matchMedia: () => ({ matches: false }) };
  context.Plotly = { react: (chart, traces) => { context.rendered = traces; } };
  vm.runInContext('DATA.cell_options = [{ id: "one", feature_cap: 10000, n_train: 100 }]; DATA.domain_elo = rows; EXCLUDED = new Set(); renderBudget("perf");', context);
  assert.equal(context.rendered[0].y[0], -10);
  assert.equal(context.rendered[0].error_y.array[0], null);
  assert.equal(context.rendered[1].y[0], 0);
  assert.equal(context.rendered[1].error_y.array[0], 40);
  assert.match(elements['performance-note'].textContent, /clipped at zero/);
});


test('embeddings order uses the higher rating and row labels have no subtitle', () => {
  const spec = plot([row('XT', 1011), row('XT-TUNED', 927), row('RF', 1000), row('RF-TUNED', 928), row('GBM', 954)]);
  assert.deepEqual(Array.from(spec.peers, row => row.model_id), ['GBM', 'RF', 'XT']);
  assert.deepEqual(Array.from(spec.layout.yaxis.ticktext), ['GBM', 'RF', 'XT']);
  assert.equal(spec.traces[0].customdata[2][4], 1011);
  assert.equal(spec.traces[0].customdata[2][5], 927);
  assert.ok(spec.traces[0].hovertemplate[2].includes('Ranking Elo (max)'));
  assert.equal(spec.layout.shapes[0].x0, 1000);
});

test('improvements, regressions and ties sort by maximum without changing ratings', () => {
  const rows = [row('RF', 1000), row('RF-TUNED', 850), row('XT', 900), row('XT-TUNED', 1100), row('LR', 975), row('LR-TUNED', 975)];
  const before = JSON.stringify(rows);
  const spec = plot(rows);
  assert.deepEqual(Array.from(spec.peers, row => row.model_id), ['LR', 'RF', 'XT']);
  assert.equal(JSON.stringify(rows), before);
  assert.deepEqual(Array.from(spec.traces[2].x), [850, 1100, 975]);
});


test('show all reveals negative Elo and untuned model curves', () => {
  const context = vm.createContext({ document: { body: { classList: { contains: () => false } } } });
  vm.runInContext(source, context);
  context.rows = [row('RF', 1000), row('DUMMY', -100)];
  context.models = Object.fromEntries(context.rows.map(r => [r.model_id, {...r, tuned_from: null}]));
  const result = vm.runInContext('DATA = {models}; SHOW_ALL_MODELS = true; eloPlotSpec(rows, false, 10000)', context);
  assert.equal(result.peers.length, 2);
  assert.equal(result.hiddenNegativeCount, 0);
  assert.equal(vm.runInContext('preferTunedRows(rows).length', context), 2);
});

test('rank stability responds to exclusions and handles tied ranks', () => {
  const context = vm.createContext({});
  vm.runInContext(source, context);
  context.rows = [
    {cell:'a', model_id:'RF', Elo:1}, {cell:'a', model_id:'CAT', Elo:2}, {cell:'a', model_id:'XT', Elo:3},
    {cell:'b', model_id:'RF', Elo:1}, {cell:'b', model_id:'CAT', Elo:3}, {cell:'b', model_id:'XT', Elo:2},
  ].map(r => ({...r, domain:'all', metric:'f1_macro'}));
  assert.equal(vm.runInContext('DATA = {domain_elo: rows}; visibleRankCorrelation("a", "b")', context), 0.5);
  assert.equal(vm.runInContext('EXCLUDED = new Set(["CAT"]); visibleRankCorrelation("a", "b")', context), 1);
});
