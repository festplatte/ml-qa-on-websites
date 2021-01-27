import { MongoClient } from "mongodb";

const url = process.env.MONGODB_CONNECTION;
let cachedClient;

const connect = () =>
  new Promise((resolve, reject) => {
    MongoClient.connect(url, (err, db) => {
      if (err) throw err;
      console.log("database connected!");
      resolve(db.db("human-eval"));
    });
  });

const mongoClient = async () => {
  if (!cachedClient) {
    cachedClient = await connect();
  }
  return cachedClient;
};

export const insert = (collection, object) =>
  new Promise(async (resolve, reject) => {
    const dbo = await mongoClient();
    dbo.collection(collection).insertOne(object, (err, result) => {
      if (err) {
        reject(err);
        return;
      }
      console.log(`1 document inserted into ${collection}`);
      resolve(result);
    });
  });

export const findOne = (collection, query = {}) =>
  new Promise(async (resolve, reject) => {
    const dbo = await mongoClient();
    dbo.collection(collection).findOne(query, (err, result) => {
      if (err) {
        reject(err);
        return;
      }
      resolve(result);
    });
  });

export const findAll = (collection, query = {}) =>
  new Promise(async (resolve, reject) => {
    const dbo = await mongoClient();
    dbo
      .collection(collection)
      .find(query)
      .toArray((err, result) => {
        if (err) {
          reject(err);
          return;
        }
        resolve(result);
      });
  });
