const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const { test } = require('node:test');
const root = path.join(__dirname, '../src/tabbench_bio/web/js');
const app = fs.readFileSync(path.join(root, 'app.js'), 'utf8').replace('main().catch(showLoadError);', '');
const datasets = fs.readFileSync(path.join(root, 'datasets.js'), 'utf8');

function context(page, search = '', identity = {}) {
  const elements = {};
  const ctx = vm.createContext({
    URL, URLSearchParams, console,
    document: { body: { dataset: { page, ...identity } }, getElementById: id =>
      elements[id] ||= { value: '', options: [], style: {setProperty() {}}, addEventListener() {},
        removeAttribute(name) { delete this[name]; } },
      querySelector: selector => elements[selector] ||= {} },
    location: { href: `https://tabbench-bio.eu/${page}.html${search}`, pathname: `/${page}.html`, search,
      replace(url) { this.replaced = url; } },
  });
  ctx.history = { replaceState(_state, _title, url) {
    const parsed = new URL(url, ctx.location.href);
    Object.assign(ctx.location, {href: parsed.href, pathname: parsed.pathname, search: parsed.search});
  } };
  ctx.window = ctx;
  vm.runInContext(app, ctx);
  return { ctx, elements };
}

test('detail URLs encode model and dataset identifiers as query parameters', () => {
  const { ctx } = context('model');
  assert.equal(vm.runInContext('detailPageUrl("model", "RF")', ctx), 'model.html?model=RF');
  assert.equal(vm.runInContext('detailPageUrl("model", "custom/α")', ctx), 'model.html?model=custom%2F%CE%B1');
  assert.equal(vm.runInContext('detailPageUrl("dataset", "x~2F?&")', ctx), 'dataset.html?id=x%7E2F%3F%26');
});

test('legacy model links redirect to the matching page and retain control settings', async () => {
  const { ctx } = context('models', '?model=RF&p=2000&compare=XT');
  ctx.fetch = async () => ({ ok: true, json: async () => ({ models: { RF: {} }, meta: { plot_excluded_models: [] } }) });
  vm.runInContext('initializeTheme = () => {}; restoreModelMetadata = () => {};', ctx);
  await vm.runInContext('main()', ctx);
  assert.equal(ctx.location.replaced, 'model.html?model=RF&p=2000&compare=XT');
});

test('dataset query URLs load directly and registry links preserve filters and budgets', async () => {
  const { ctx, elements } = context('dataset', '?id=GEO-X&p=2000&metric=f1_macro');
  vm.runInContext(datasets, ctx);
  ctx.fetch = async url => {
    assert.equal(url, 'data/datasets/scores.json');
    return {ok: true, json: async () => ({dataset_id: 'GEO-X', scores: []})};
  };
  vm.runInContext(`DATA = {
    datasets: [{dataset_id: 'GEO-X', display_name: 'Example dataset', modality: 'Gene expression',
      task: 'Classification', target: 'Diagnosis', source: 'geo', source_url: null,
      scores_file: 'scores.json', default_cell: 'ref', metrics: [{key: 'f1_macro', label: 'Macro-F1'}]}],
    models: {}, cell_options: [{id: 'ref', feature_cap: 2000, n_train: 100}]
  };
  setOptions = (select, options, value) => { select.value = String(value); };
  datasetCells = () => DATA.cell_options;
  loadExternalScript = async () => {};
  renderDatasetCharts = () => {};`, ctx);
  await vm.runInContext('initializeDatasetDetail()', ctx);
  assert.equal(ctx.location.replaced, undefined);
  assert.equal(ctx.document.title, 'Example dataset · TabBench-Bio');
  assert.equal(elements['link[rel="canonical"]'].href, 'https://tabbench-bio.eu/dataset.html?id=GEO-X');
  assert.equal(vm.runInContext('datasetLink("GEO-Y")', ctx), 'dataset.html?id=GEO-Y&p=2000&metric=f1_macro');
});

test('model switching updates the URL, title, canonical and card while retaining controls', () => {
  const { ctx, elements } = context('model', '?model=RF&p=2000&compare=XT');
  vm.runInContext(`
    DATA = { models: {RF: { id: 'RF', display: 'Random Forest', category: 'Tree-based',
        source_url: 'https://github.com/scikit-learn/scikit-learn' },
        XT: { id: 'XT', display: 'Extra Trees', category: 'Tree-based', source_url: null }},
      domains: [], domain_elo: [], model_card_coverage: [],
      cell_options: [{id: 'ref', feature_cap: 2000}], meta: {reference_cell: 'ref'}, reference: [] };
    setOptions = (select, options, current) => { select.value = String(current); };
    initializeModelCard();
  `, ctx);
  assert.equal(elements['model-card-model'].value, 'RF');
  assert.equal(elements['model-card-comparator'].value, 'XT');
  assert.equal(elements['model-card-cap'].value, '2000');
  assert.equal(elements['model-card-source'].href, 'https://github.com/scikit-learn/scikit-learn');
  assert.equal(elements['model-card-source'].hidden, false);
  elements['model-card-model'].value = 'XT';
  elements['model-card-model'].onchange();
  assert.equal(ctx.location.search, '?model=XT&p=2000&compare=XT');
  assert.equal(elements['model-card-source'].hidden, true);
  assert.equal(elements['model-card-source'].href, undefined);
  assert.equal(ctx.document.title, 'Extra Trees · TabBench-Bio');
  assert.equal(elements['model-card-name'].textContent, 'Extra Trees');
  assert.equal(elements['link[rel="canonical"]'].href, 'https://tabbench-bio.eu/model.html?model=XT');
  elements['model-card-model'].value = 'RF';
  elements['model-card-model'].onchange();
  assert.equal(ctx.location.search, '?model=RF&p=2000&compare=XT');
  vm.runInContext('delete DATA.models.RF.source_url', ctx);
  assert.throws(() => elements['model-card-model'].onchange(), /Missing source_url for model RF/);
});

test('missing reference results are not replaced by the first model', () => {
  const { ctx } = context('model');
  vm.runInContext(`DATA = {
    models: {RF: {id: 'RF', display: 'Random Forest', source_url: null}}, reference: []
  };`, ctx);
  assert.throws(() => vm.runInContext('initializeModelCard()', ctx), /model_id/);
});
