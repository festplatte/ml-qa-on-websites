const basePath = "/api";

export const getTemplates = async () => {
  const result = await fetch(`${basePath}/templates`, { method: "GET" });
  return await result.json();
};

export const saveVoting = async (voting) => {
  const result = await fetch(`${basePath}/votings`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(voting),
  });
  return await result.json();
};
