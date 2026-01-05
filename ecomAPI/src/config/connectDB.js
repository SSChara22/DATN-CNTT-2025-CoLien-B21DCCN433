const { Sequelize } = require("sequelize");
const path = require("path");

require("dotenv").config({ path: path.resolve(__dirname, "../../.env") });

console.log("Environment variables loaded:");
console.log("DB_HOST:", process.env.DB_HOST);
console.log("DB_USERNAME:", process.env.DB_USERNAME);
console.log("DB_PASSWORD:", process.env.DB_PASSWORD ? `"${process.env.DB_PASSWORD}"` : "undefined");
console.log("DB_DATABASE_NAME:", process.env.DB_DATABASE_NAME);
console.log("DB_PORT:", process.env.DB_PORT);
// Option 2: Passing parameters separately (other dialects)
const sequelize = new Sequelize(
    process.env.DB_DATABASE_NAME,
    process.env.DB_USERNAME,
    process.env.DB_PASSWORD !== undefined ? process.env.DB_PASSWORD : '',  // explicit check for undefined
    {
        host: process.env.DB_HOST,
        port: process.env.DB_PORT,
        dialect: "mysql",
        logging: false,
        dialectOptions:
            process.env.DB_SSL === "true"
                ? {
                      ssl: {
                          require: true,
                          rejectUnauthorized: false,
                      },
                  }
                : {},
        query: {
            raw: true,
        },
        timezone: "+07:00",
    }
);

let connectDB = async () => {
    try {
        await sequelize.authenticate();
        console.log("Connection has been established successfully.");
    } catch (error) {
        console.error("Unable to connect to the database:", error);
    }
};

module.exports = connectDB;
