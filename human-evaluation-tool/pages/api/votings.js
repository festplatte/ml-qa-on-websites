import { findAll, insert } from "../../utils/mongo";

const collection = "votings";

export default async (req, res) => {
  const { method } = req;
  let result;
  switch (method) {
    case "GET":
      result = await findAll(collection);
      res.status(200).json(result);
      break;
    case "POST":
      if (req.body._id) {
        delete req.body._id;
      }
      result = await insert(collection, req.body);
      res.status(201).json(result);
      break;
    default:
      res.setHeader("Allow", ["GET", "POST"]);
      res.status(405).end(`Method ${method} Not Allowed`);
  }
};
