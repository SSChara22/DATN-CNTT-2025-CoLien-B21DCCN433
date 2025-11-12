'use strict';

module.exports = {
    up: async (queryInterface, Sequelize) => {
        await queryInterface.createTable('Interactions', {
            interId: {
                allowNull: false,
                autoIncrement: true,
                primaryKey: true,
                type: Sequelize.INTEGER
            },
            userId: {
                type: Sequelize.INTEGER,
                allowNull: false,
                references: {
                    model: 'Users',
                    key: 'id'
                },
                onUpdate: 'CASCADE',
                onDelete: 'CASCADE'
            },
            productId: {
                type: Sequelize.INTEGER,
                allowNull: false,
                references: {
                    model: 'Products',
                    key: 'id'
                },
                onUpdate: 'CASCADE',
                onDelete: 'CASCADE'
            },
            actionId: {
                type: Sequelize.INTEGER,
                allowNull: false,
                references: {
                    model: 'Allcodes', // liên kết với Allcode
                    key: 'id'
                },
                onUpdate: 'CASCADE',
                onDelete: 'RESTRICT'
            },
            device_type: {
                type: Sequelize.STRING(50),
                allowNull: true
            },
            timestamp: {
                type: Sequelize.DATE,
                allowNull: false,
                defaultValue: Sequelize.literal('CURRENT_TIMESTAMP')
            }
        });

        // Ràng buộc unique để "ghi đè" hành động cũ của cùng user & product
        await queryInterface.addConstraint('Interactions', {
            fields: ['userId', 'productId'],
            type: 'unique',
            name: 'unique_user_product'
        });
    },

    down: async (queryInterface, Sequelize) => {
        await queryInterface.dropTable('Interactions');
    }
};
