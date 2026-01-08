import db from "../models/index";
import fs from "fs";
import path from "path";
import moment from "moment";
const { Op } = db.Sequelize;

// Optional Python inference bridge (default export)
let pythonInvoker = null;
try {
  pythonInvoker = require("./pythonInvoker.js").default;
} catch (e) {
  pythonInvoker = null;
}
const ROOT = path.resolve(__dirname, "../../..");
const PERF_CSV = path.join(ROOT, "processed_data", "model_performance.csv");

function pickBestModel() {
  try {
    const csv = fs.readFileSync(PERF_CSV, "utf8");
    const lines = csv.trim().split(/\r?\n/).slice(1);
    let best = { name: null, map10: -1 };
    for (const line of lines) {
      const [Model, , MAP10] = line.split(",");
      const map = parseFloat(MAP10);
      if (!isNaN(map) && map > best.map10) best = { name: Model, map10: map };
    }
    return best.name || "LNCM";
  } catch {
    // When no CSV found, we still expose a model name label for UI
    return "LNCM";
  }
}

async function ensureTables() {
  const qi = db.sequelize.getQueryInterface();
  // recommendations
  try {
    await qi.describeTable('recommendations');
  } catch {
    await qi.createTable('recommendations', {
      id: { type: db.Sequelize.INTEGER, primaryKey: true, autoIncrement: true, allowNull: false },
      userId: { type: db.Sequelize.INTEGER, allowNull: false },
      productId: { type: db.Sequelize.INTEGER, allowNull: false },
      modelName: { type: db.Sequelize.STRING(50), allowNull: false },
      score: { type: db.Sequelize.FLOAT, allowNull: false, defaultValue: 0 },
      details: { type: db.Sequelize.TEXT('long'), allowNull: true },
      createdAt: { type: db.Sequelize.DATE, allowNull: false, defaultValue: db.Sequelize.NOW },
      updatedAt: { type: db.Sequelize.DATE, allowNull: false, defaultValue: db.Sequelize.NOW }
    });
  }
  // model_runs
  try {
    await qi.describeTable('model_runs');
  } catch {
    await qi.createTable('model_runs', {
      id: { type: db.Sequelize.INTEGER, primaryKey: true, autoIncrement: true, allowNull: false },
      userId: { type: db.Sequelize.INTEGER, allowNull: false },
      modelName: { type: db.Sequelize.STRING(50), allowNull: false },
      metricsJson: { type: db.Sequelize.TEXT('long'), allowNull: true },
      recommendationsJson: { type: db.Sequelize.TEXT('long'), allowNull: true },
      createdAt: { type: db.Sequelize.DATE, allowNull: false, defaultValue: db.Sequelize.NOW },
      updatedAt: { type: db.Sequelize.DATE, allowNull: false, defaultValue: db.Sequelize.NOW }
    });
  }
}

function deriveContext(ts) {
  const m = moment(ts);
  const hour = m.hour();
  const month = m.month() + 1;
  const dow = m.isoWeekday();
  const isWeekend = dow >= 6 ? 1 : 0;
  const season = month <= 2 || month === 12 ? "winter" : month <= 5 ? "spring" : month <= 8 ? "summer" : "autumn";
  const time_of_day = hour < 6 ? "night" : hour < 12 ? "morning" : hour < 18 ? "afternoon" : "evening";
  return { hour, month, day_of_week: dow, is_weekend: isWeekend, season, time_of_day, day_name: m.format("dddd"), date: m.format("YYYY-MM-DD") };
}

function priceRange(price) {
  if (price == null) return "unknown";
  if (price < 200000) return "low";
  if (price < 1000000) return "mid";
  if (price < 3000000) return "high";
  return "premium";
}

function intentFromInteractions(actions) {
  // high if any purchase, medium if any cart, else low
  const set = new Set(actions);
  return set.has("purchase") ? "high" : set.has("cart") ? "medium" : "low";
}

function scoreSample({ rating, rating_count, product_views, discount_percentage, intent_weight }) {
  // Simple bounded 0..1 score
  const r = isNaN(rating) ? 0 : Math.max(0, Math.min(1, (rating - 1) / 4));
  const rc = Math.min(1, (rating_count || 0) / 50);
  const pv = Math.min(1, (product_views || 0) / 50);
  const disc = Math.min(1, Math.max(0, (discount_percentage || 0) / 70));
  const iw = intent_weight; // 0.0, 0.5, 1.0
  return 0.45 * r + 0.15 * rc + 0.15 * pv + 0.15 * disc + 0.10 * iw;
}

async function buildUserProductFeatures(userId) {
  // gather products and features
  const products = await db.Product.findAll({
    where: { statusId: 'S1' },
    raw: true
  });
  const productIds = products.map(p => p.id);
  // map product -> details
  const details = await db.ProductDetail.findAll({ where: { productId: { [Op.in]: productIds } }, raw: true });
  const detailByProduct = new Map();
  for (const d of details) {
    if (!detailByProduct.has(d.productId)) detailByProduct.set(d.productId, []);
    detailByProduct.get(d.productId).push(d);
  }
  const comments = await db.Comment.findAll({ where: { productId: { [Op.in]: productIds } }, raw: true });
  const commentsByProduct = new Map();
  for (const c of comments) {
    if (!commentsByProduct.has(c.productId)) commentsByProduct.set(c.productId, []);
    commentsByProduct.get(c.productId).push(c);
  }
  const interactions = await db.Interaction.findAll({ where: { userId, productId: { [Op.in]: productIds } }, raw: true, order: [['timestamp','DESC']] });
  const lastInter = interactions[0];
  const actsByProduct = new Map();
  for (const it of interactions) {
    if (!actsByProduct.has(it.productId)) actsByProduct.set(it.productId, []);
    actsByProduct.get(it.productId).push(it.actionCode);
  }
  // derive user-level context
  const nowCtx = deriveContext(new Date());
  const user = await db.User.findOne({ where: { id: userId }, raw: true });
  const genderCode = user?.genderId || null;
  // device type: use last interaction else unknown
  const device_type = lastInter?.device_type || 'unknown';

  const results = [];
  for (const p of products) {
    const det = (detailByProduct.get(p.id) || [])[0] || {};
    const price = det.discountPrice || null;
    const orig = det.originalPrice || null;
    const discount_percentage = price && orig ? Math.round(((orig - price) / Math.max(1, orig)) * 100) : 0;
    const product_views = p.view || 0;
    const list = commentsByProduct.get(p.id) || [];
    const rating_count = list.length;
    const rating = rating_count ? (list.reduce((s, c) => s + (c.star || 0), 0) / rating_count) : null;
    const actions = actsByProduct.get(p.id) || [];
    const purchase_intent = intentFromInteractions(actions);
    const intent_weight = purchase_intent === 'high' ? 1.0 : purchase_intent === 'medium' ? 0.5 : 0.0;
    const s = scoreSample({ rating, rating_count, product_views, discount_percentage, intent_weight });
    // build full feature record for audit/comparison if needed
    const ctx = nowCtx; // using current-time context; per-interaction context can be added if desired
    results.push({
      productId: p.id,
      price, original_price: orig,
      discount_percentage,
      product_views,
      rating: rating || 0,
      rating_count,
      purchase_intent,
      interaction_type: actions[0] || null,
      timestamp: lastInter?.timestamp || null,
      device_type,
      gender: genderCode || null,
      category: p.categoryId,
      brand: p.brandId,
      price_range: priceRange(price),
      time_of_day: ctx.time_of_day,
      day_of_week: ctx.day_of_week,
      month: ctx.month,
      season: ctx.season,
      day_name: ctx.day_name,
      is_weekend: ctx.is_weekend,
      hour: ctx.hour,
      date: ctx.date,
      score: s
    });
  }
  // sort by score desc
  results.sort((a,b)=>b.score-a.score);
  return results;
}


async function computeRecommendationsForUser(userId, limit=10) {
  // Early role gate: only allow users with roleId 'R2'
  const userRow = await db.User.findOne({ where: { id: userId }, raw: true });
  if (!userRow) {
    return { bestModel: null, top: [], modelRuns: [], error: 'User not found' };
  }
  if ((userRow.roleId || '').toUpperCase() !== 'R2') {
    return { bestModel: null, top: [], modelRuns: [], error: 'User role not permitted for recommendations' };
  }
  // Try Python inference first: evaluate trained models and select best by MAP@10 then Precision@10
  // Ground truth includes both cart and purchase interactions for better evaluation
  if (pythonInvoker) {
    // Use both purchase and cart as ground truth for evaluation
    const gtInteractions = await db.Interaction.findAll({
      where: {
        userId,
        actionCode: { [Op.in]: ['purchase', 'cart'] }
      },
      attributes: ['productId'],
      raw: true
    });
    const gtSet = new Set(gtInteractions.map(r => r.productId));
    const nowCtx = deriveContext(new Date());
    const lastInter = await db.Interaction.findOne({ where: { userId }, order: [['timestamp','DESC']], raw: true });

    // Get user information for richer context
    const user = await db.User.findOne({ where: { id: userId }, raw: true });
    const gender = user?.genderId || null;

    // Get user's interaction history for preferences
    const userInteractions = await db.Interaction.findAll({
      where: { userId, actionCode: { [Op.in]: ['cart', 'purchase'] } },
      include: [{ model: db.Product, as: 'productData', attributes: ['categoryId', 'brandId'] }],
      raw: true, nest: true, limit: 50, order: [['timestamp', 'DESC']]
    });

    // Extract preferred categories and brands
    const categoryPrefs = new Set();
    const brandPrefs = new Set();
    for (const inter of userInteractions) {
      if (inter.productData?.categoryId) categoryPrefs.add(inter.productData.categoryId);
      if (inter.productData?.brandId) brandPrefs.add(inter.productData.brandId);
    }

    const ctxPayload = {
      time_of_day: nowCtx.time_of_day,
      season: nowCtx.season,
      device_type: lastInter?.device_type || 'unknown',
      gender: gender,
      preferred_categories: Array.from(categoryPrefs),
      preferred_brands: Array.from(brandPrefs),
      category: null  // kept for backward compatibility
    };

    // Run all models for evaluation and selection
    const modelNames = ['ENCM','LNCM','NeuMF','BMF'];
    const k = Math.max(1, Math.min(10, limit));

    // Run the 4 model inferences sequentially
    const parallel = [];
    for (const name of modelNames) {
      try {
        // [DISABLED CLIENT-SIDE SEND]
        // Re-enable by removing this conditional block.
        if (['BMF','NeuMF','LNCM'].includes(name)) {
          console.log(`[RECO] Skipping Python call for disabled model ${name}`);
          parallel.push({ name, resp: { ok: false, error: 'disabled_client_side' } });
          continue;
        }
        console.log(`[RECO] Calling model ${name}...`);
        const resp = await pythonInvoker.runPythonInference({ user_id: userId, limit: k, model: name, context: ctxPayload });
        console.log(`[RECO] Model ${name} response:`, {
          ok: resp.ok,
          hasItems: resp.items && Array.isArray(resp.items),
          itemsLength: resp.items ? resp.items.length : 0,
          error: resp.error
        });
        parallel.push({ name, resp });
      } catch (e) {
        console.log(`[RECO] Model ${name} exception:`, e);
        parallel.push({ name, resp: { ok: false, error: e?.message || String(e) } });
      }
    }

    console.log(`[RECO] All responses:`, parallel.map(p => ({ name: p.name, ok: p.resp.ok, error: p.resp.error })));
    const modelRuns = [];
    for (const { name, resp } of parallel) {
      let recs = [];
      if (resp && resp.ok && Array.isArray(resp.items)) {
        recs = resp.items.map(it => ({ productId: it.productId, score: it.score || 0 }));
      }
      let hits = 0; let sumPrec = 0;
      const denom = Math.max(1, Math.min(k, gtSet.size || 0));
      for (let i = 0; i < Math.min(k, recs.length); i++) {
        const rel = gtSet.has(recs[i].productId) ? 1 : 0;
        if (rel) { hits += 1; sumPrec += hits / (i + 1); }
      }
      const precision10 = recs.length ? (hits / k) : 0;
      const map10 = denom ? (sumPrec / denom) : 0;
      modelRuns.push({ modelName: name, recommendations: recs.slice(0, k), metrics: { mode: 'python_infer', precision10, map10 } });
    }
    modelRuns.sort((a,b)=> (b.metrics.map10 - a.metrics.map10) || (b.metrics.precision10 - a.metrics.precision10));
    if (modelRuns.length) {
      const best = modelRuns[0];
      return { bestModel: best.modelName, top: best.recommendations, modelRuns };
    }
    // fall through to heuristic if Python failed
  } else {
    console.log(`[RECO] Python invoker not available, falling back to heuristic`);
  }

  console.log(`[RECO] Using heuristic recommendation approach`);

  // Generate multiple DB-scoring variants and choose best
  const perModelLimit = 10; // each model should output 10 items
  const base = await buildUserProductFeatures(userId);

  // Use same ground truth as Python evaluation
  const gtInteractions = await db.Interaction.findAll({
    where: {
      userId,
      actionCode: { [Op.in]: ['purchase', 'cart'] }
    },
    attributes: ['productId'],
    raw: true
  });
  const gtSet = new Set(gtInteractions.map(r => r.productId));

  const likedInter = await db.Interaction.findAll({
    where: { userId, actionCode: { [Op.in]: ['cart','purchase'] } },
    include: [{ model: db.Product, as: 'productData', attributes: ['id','categoryId','brandId'] }],
    raw: true, nest: true
  });
  const catSet = new Set(); const brandSet = new Set();
  for (const it of likedInter) { if (it.productData?.categoryId) catSet.add(it.productData.categoryId); if (it.productData?.brandId) brandSet.add(it.productData.brandId); }

  const variants = [
    // Names aligned to requested set: bmf, encm, lncm, neumf
    { name: 'bmf',   weights: { r:0.20, rc:0.10, pv:0.20, disc:0.35, iw:0.15 } },
    { name: 'encm',  weights: { r:0.30, rc:0.10, pv:0.15, disc:0.10, iw:0.35 } },
    { name: 'lncm',  weights: { r:0.55, rc:0.20, pv:0.10, disc:0.10, iw:0.05 } },
    { name: 'neumf', weights: { r:0.40, rc:0.15, pv:0.15, disc:0.10, iw:0.20 } },
  ];

  function rescore(list, w){
    return list.map(x=>{
      const r = Math.max(0, Math.min(1, (x.rating - 1) / 4));
      const rc = Math.min(1, (x.rating_count||0)/50);
      const pv = Math.min(1, (x.product_views||0)/50);
      const disc = Math.min(1, Math.max(0, (x.discount_percentage||0)/70));
      const iw = x.purchase_intent==='high'?1.0:x.purchase_intent==='medium'?0.5:0.0;
      const score = w.r*r + w.rc*rc + w.pv*pv + w.disc*disc + w.iw*iw;
      return { ...x, score };
    }).sort((a,b)=>b.score-a.score)
  }

  async function backfillTop(scoredItems){
    let top = scoredItems.slice(0, perModelLimit).map(x=>({ productId:x.productId, score:x.score }));
    if (top.length >= perModelLimit) return top;
    const existing = new Set(top.map(t=>t.productId));
    if (catSet.size || brandSet.size){
      const sameCatOrBrand = await db.Product.findAll({
        where: {
          statusId:'S1', id:{ [Op.notIn]: Array.from(existing) },
          [Op.or]: [
            catSet.size? { categoryId: { [Op.in]: Array.from(catSet) } } : null,
            brandSet.size? { brandId: { [Op.in]: Array.from(brandSet) } } : null,
          ].filter(Boolean)
        }, order: [['view','DESC']], attributes:['id'], raw:true
      });
      for (const p of sameCatOrBrand){ if (!existing.has(p.id)){ top.push({productId:p.id, score:0.0}); existing.add(p.id);} if (top.length===perModelLimit) break; }
    }
    if (top.length < perModelLimit){
      const need = perModelLimit-top.length;
      const popular = await db.Product.findAll({ where:{ statusId:'S1', id:{ [Op.notIn]: Array.from(existing) } }, order:[['view','DESC']], limit: need, attributes:['id'], raw:true });
      for (const p of popular){ if (!existing.has(p.id)){ top.push({productId:p.id, score:0.0}); existing.add(p.id);} if (top.length===perModelLimit) break; }
    }
    return top;
  }

  const modelRuns = [];
  for (const v of variants){
    const rescored = rescore(base, v.weights);
    const topV = await backfillTop(rescored);
    // simple metrics: avg score, intent hits, cat/brand alignment
    const byId = new Map(rescored.map(x=>[x.productId, x]));
    const avgScore = topV.reduce((s,t)=>s+(byId.get(t.productId)?.score||0),0)/Math.max(1, topV.length);
    let aligned = 0; let intentHigh=0;
    for (const t of topV){
      const item = byId.get(t.productId);
      if (!item) continue;
      if (catSet.has(item.category) || brandSet.has(item.brand)) aligned++;
      if (item.purchase_intent==='high') intentHigh++;
    }
    modelRuns.push({ modelName: v.name, recommendations: topV, metrics: { mode:'db_scoring', avgScore, aligned, intentHigh } });
  }

  // choose best: prioritize intentHigh, then aligned, then avgScore
  modelRuns.sort((a,b)=> (b.metrics.intentHigh - a.metrics.intentHigh) || (b.metrics.aligned - a.metrics.aligned) || (b.metrics.avgScore - a.metrics.avgScore));
  const best = modelRuns[0];
  return { bestModel: best.modelName, top: best.recommendations, modelRuns };

}

// In-process guard to prevent concurrent recomputations per user
const _recomputeInProgress = new Set();

async function initForUser(userId, limit=10) {
  // Early role gate: only allow users with roleId 'R2'
  const userRow = await db.User.findOne({ where: { id: userId }, raw: true });
  if (!userRow) {
    return { errCode: 1, message: 'User not found' };
  }
  if ((userRow.roleId || '').toUpperCase() !== 'R2') {
    return { errCode: 2, message: 'User role not permitted for recommendations' };
  }
  // Skip if a recomputation is already running for this user (debounce concurrent triggers)
  if (_recomputeInProgress.has(userId)) {
    return { errCode: 0, message: 'recompute_in_progress' };
  }
  _recomputeInProgress.add(userId);
  const t = await db.sequelize.transaction();
  await ensureTables();
  try {
    // clear previous inside transaction
    await db.Recommendation.destroy({ where: { userId }, transaction: t });
    await db.ModelRun.destroy({ where: { userId }, transaction: t });
    const { bestModel, top, modelRuns, error } = await computeRecommendationsForUser(userId, limit);
    if (error) {
      await t.rollback();
      _recomputeInProgress.delete(userId);
      return { errCode: 3, message: error };
    }
    // save cache (bulk, ignore duplicates for safety)
    if (top && top.length) {
      const recRows = top.map(item => ({ userId, productId: item.productId, modelName: bestModel, score: item.score || 0, details: null }));
      await db.Recommendation.bulkCreate(recRows, { ignoreDuplicates: true, transaction: t });
    }
    if (modelRuns && modelRuns.length) {
      const runRows = modelRuns.map(r => ({ userId, modelName: r.modelName, metricsJson: JSON.stringify(r.metrics||{}), recommendationsJson: JSON.stringify(r.recommendations||[]) }));
      await db.ModelRun.bulkCreate(runRows, { ignoreDuplicates: true, transaction: t });
    }
    await t.commit();
    return { bestModel };
  } catch (e) {
    try { await t.rollback(); } catch {}
    return { errCode: 3, message: e?.message || String(e) };
  } finally {
    _recomputeInProgress.delete(userId);
  }
}

async function getCachedForUser(userId, limit=10) {
  await ensureTables();
  const rows = await db.Recommendation.findAll({ where: { userId }, order: [['score', 'DESC']], limit, raw: true });
  return rows;
}

async function clearForUser(userId) {
  await ensureTables();
  await db.Recommendation.destroy({ where: { userId } });
  await db.ModelRun.destroy({ where: { userId } });
  return true;
}

module.exports = { initForUser, getCachedForUser, clearForUser };