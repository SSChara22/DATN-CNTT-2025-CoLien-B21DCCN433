import recommendationService from "../services/recommendationService";
import db from "../models";
const { Op } = db.Sequelize;

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

    // Hydrate product names for model runs
    const runsWithProducts = [];
    for (const run of runs) {
      try {
        const recs = JSON.parse(run.recommendationsJson || '[]');
        const recsWithNames = await Promise.all(recs.map(async (r) => {
          const product = await db.Product.findOne({ where: { id: r.productId } });
          return {
            ...r,
            productName: product ? product.name : 'Unknown Product',
            brandName: product ? product.brandId : 'Unknown Brand'
          };
        }));
        runsWithProducts.push({
          ...run,
          recommendationsJson: JSON.stringify(recsWithNames)
        });
      } catch (e) {
        // If parsing fails, keep original
        runsWithProducts.push(run);
      }
    }

    // Hydrate product names for cached recommendations
    const cachedWithProducts = [];
    for (const r of cached) {
      const product = await db.Product.findOne({ where: { id: r.productId } });
      cachedWithProducts.push({
        ...r,
        productName: product ? product.name : 'Unknown Product',
        brandName: product ? product.brandId : 'Unknown Brand'
      });
    }

    // Get user context for display
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

    // Get last interaction for device type
    const lastInter = await db.Interaction.findOne({ where: { userId }, order: [['timestamp','DESC']], raw: true });
    const device_type = lastInter?.device_type || 'unknown';

    // Current time context
    const now = new Date();
    const hour = now.getHours();
    const month = now.getMonth() + 1;

    // Derive context features
    const time_of_day = hour < 6 ? 0 : (hour < 12 ? 1 : (hour < 18 ? 2 : 3));  // night, morning, afternoon, evening
    const season = month <= 2 || month === 12 ? 0 : (month <= 5 ? 1 : (month <= 8 ? 2 : 3));  // winter, spring, summer, autumn

    // Gender encoding
    const genderMap = {'M': 0, 'FE': 1, 'O': 2};
    const gender_code = genderMap[gender] !== undefined ? genderMap[gender] : 3;  // 3 for unknown/other

    const context = {
      gender: gender_code,
      device_type: device_type,
      time_of_day: time_of_day,
      season: season,
      month: month,
      hour: hour,
      preferred_categories: Array.from(categoryPrefs),
      preferred_brands: Array.from(brandPrefs)
    };

    let bestModelName = null;
    if (cachedWithProducts && cachedWithProducts.length) bestModelName = cachedWithProducts[0].modelName || null;
    res.render('recommend_dashboard', {
      runs: runsWithProducts,
      cached: cachedWithProducts,
      userId,
      bestModelName,
      countCached: cachedWithProducts.length,
      countRuns: runsWithProducts.length,
      context: JSON.stringify(context)
    });
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
