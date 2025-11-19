'use strict';
module.exports = (sequelize, DataTypes) => {
  const ModelRun = sequelize.define('ModelRun', {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
    userId: { type: DataTypes.INTEGER, allowNull: false },
    modelName: { type: DataTypes.STRING(50), allowNull: false },
    metricsJson: { type: DataTypes.TEXT('long'), allowNull: true },
    recommendationsJson: { type: DataTypes.TEXT('long'), allowNull: true },
    createdAt: { type: DataTypes.DATE, allowNull: false, defaultValue: DataTypes.NOW },
    updatedAt: { type: DataTypes.DATE, allowNull: false, defaultValue: DataTypes.NOW }
  }, {
    tableName: 'model_runs',
    timestamps: true
  });
  ModelRun.associate = function(models) {
    ModelRun.belongsTo(models.User, { foreignKey: 'userId', as: 'user' });
  };
  return ModelRun;
};
