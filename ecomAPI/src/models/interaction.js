'use strict';
const { Model } = require('sequelize');

module.exports = (sequelize, DataTypes) => {
    class Interaction extends Model {
        static associate(models) {
            Interaction.belongsTo(models.User, { foreignKey: 'userId', as: 'userData' });
            Interaction.belongsTo(models.Product, { foreignKey: 'productId', as: 'productData' });
            // FK tới Allcode bằng cột 'code' thay vì 'id'
            Interaction.belongsTo(models.Allcode, { foreignKey: 'actionCode', targetKey: 'code', as: 'actionData' });
        }
    }

    Interaction.init(
        {
            interId: {
                allowNull: false,
                autoIncrement: true,
                primaryKey: true,
                type: DataTypes.INTEGER
            },
            userId: {
                type: DataTypes.INTEGER,
                allowNull: false
            },
            productId: {
                type: DataTypes.INTEGER,
                allowNull: false
            },
            actionCode: {
                type: DataTypes.STRING(50),
                allowNull: false
            },
            device_type: {
                type: DataTypes.STRING(50),
                allowNull: true
            },
            timestamp: {
                type: DataTypes.DATE,
                allowNull: false,
                defaultValue: DataTypes.NOW
            }
        },
        {
            sequelize,
            modelName: 'Interaction',
            tableName: 'interactions',
            timestamps: false
        }
    );

    return Interaction;
};
