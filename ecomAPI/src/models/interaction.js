'use strict';
const { Model } = require('sequelize');

module.exports = (sequelize, DataTypes) => {
    class Interaction extends Model {
        static associate(models) {
            // Liên kết với User
            Interaction.belongsTo(models.User, { foreignKey: 'userId', as: 'userData' });
            // Liên kết với Product
            Interaction.belongsTo(models.Product, { foreignKey: 'productId', as: 'productData' });
            // Liên kết với Allcode (action)
            Interaction.belongsTo(models.Allcode, { foreignKey: 'actionId', as: 'actionData' });
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
            actionId: {
                type: DataTypes.INTEGER,
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
            tableName: 'Interactions',
            timestamps: false,
            indexes: [
                {
                    unique: true,
                    fields: ['userId', 'productId']
                }
            ]
        }
    );

    return Interaction;
};
