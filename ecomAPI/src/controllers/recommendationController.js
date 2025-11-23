import recommendationService from "../services/recommendationService";
import db from "../models";

let initForCurrentUser = async (req, res) => {
  try {
    const userId = req.user.id;
    const limit = +(req.query.limit || 10);
    await recommendationService.initForUser(userId, limit);
    return res.status(200).json({ errCode: 0, message: 'initialized' });
  } catch (e) {
    return res.status(200).json({ errCode: -1, errMessage: 'Error from server' });
  }
};

let listForCurrentUser = async (req, res) => {
  try {
    const userId = req.user.id;
    const limit = +(req.query.limit || 10);
    const recs = await recommendationService.getCachedForUser(userId, limit);
    // hydrate products
    const result = [];
    for (const r of recs) {
      const product = await db.Product.findOne({ where: { id: r.productId } });
      if (product) result.push({ product, score: r.score, modelName: r.modelName });
    }
    return res.status(200).json({ errCode: 0, data: result });
  } catch (e) {
    return res.status(200).json({ errCode: -1, errMessage: 'Error from server' });
  }
};

let dashboardPage = async (req, res) => {
  try {
    // CORS for standalone viewer
    res.set('Access-Control-Allow-Origin', '*');
    res.set('Access-Control-Allow-Headers', 'Authorization, Content-Type');
    res.set('Vary', 'Origin');
    const userId = req.user.id;
    const runs = await db.ModelRun.findAll({ where: { userId }, order: [['createdAt','DESC']] });
    const cached = await db.Recommendation.findAll({ where: { userId }, order: [['score','DESC']] });
    let bestModelName = null;
    if (cached && cached.length) bestModelName = cached[0].modelName || null;
    res.render('recommend_dashboard', { runs, cached, userId, bestModelName, countCached: cached.length, countRuns: runs.length });
  } catch (e) {
    res.set('Access-Control-Allow-Origin', '*');
    res.set('Access-Control-Allow-Headers', 'Authorization, Content-Type');
    res.status(500).send('Internal Error: ' + (e?.message || e));
  }
};

module.exports = { initForCurrentUser, listForCurrentUser, dashboardPage };
 
let clearForCurrentUser = async (req, res) => {
  try {
    const userId = req.user.id;
    await recommendationService.clearForUser(userId);
    return res.status(200).json({ errCode: 0, message: 'cleared' });
  } catch (e) {
    return res.status(200).json({ errCode: -1, errMessage: 'Error from server' });
  }
};

module.exports.clearForCurrentUser = clearForCurrentUser;
