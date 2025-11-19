'use strict';
module.exports = (sequelize, DataTypes) => {
  const Recommendation = sequelize.define('Recommendation', {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
    userId: { type: DataTypes.INTEGER, allowNull: false },
    productId: { type: DataTypes.INTEGER, allowNull: false },
    modelName: { type: DataTypes.STRING(50), allowNull: false },
    score: { type: DataTypes.FLOAT, allowNull: false },
    details: { type: DataTypes.TEXT('long'), allowNull: true },
    createdAt: { type: DataTypes.DATE, allowNull: false, defaultValue: DataTypes.NOW },
    updatedAt: { type: DataTypes.DATE, allowNull: false, defaultValue: DataTypes.NOW }
  }, {
    tableName: 'recommendations',
    timestamps: true
  });
  Recommendation.associate = function(models) {
    Recommendation.belongsTo(models.User, { foreignKey: 'userId', as: 'user' });
    Recommendation.belongsTo(models.Product, { foreignKey: 'productId', as: 'product' });
  };
  return Recommendation;
};
