import { findAll, insert } from "../../utils/mongo";

const collection = "templates";

export default async (req, res) => {
  const { method } = req;
  let result;
  switch (method) {
    case "GET":
      result = await findAll(collection);
      res.status(200).json(result);
      break;
    case "POST":
      console.log(req.headers);
      if (req.headers["authorization"] === "human-eval") {
        result = await insert(collection, req.body);
        res.status(201).json(result);
      }
      res.status(401).end("Unauthorized");
      break;
    default:
      res.setHeader("Allow", ["GET", "POST"]);
      res.status(405).end(`Method ${method} Not Allowed`);
  }
};
