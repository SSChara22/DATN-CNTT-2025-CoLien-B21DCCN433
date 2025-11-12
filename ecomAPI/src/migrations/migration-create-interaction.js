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
        references: { model: 'Users', key: 'id' },
        onUpdate: 'CASCADE',
        onDelete: 'CASCADE'
      },
      productId: {
        type: Sequelize.INTEGER,
        allowNull: false,
        references: { model: 'Products', key: 'id' },
        onUpdate: 'CASCADE',
        onDelete: 'CASCADE'
      },
      actionCode: {
        type: Sequelize.STRING(50), // varchar để lưu code
        allowNull: false,
        references: { model: 'Allcodes', key: 'code' }, // FK tới code
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

  },

  down: async (queryInterface, Sequelize) => {
    await queryInterface.dropTable('Interactions');
  }
};
