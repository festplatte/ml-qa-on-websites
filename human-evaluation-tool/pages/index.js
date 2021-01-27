import React, { useEffect, useState } from "react";
import FinalScreen from "../components/FinalScreen";
import QuestionScreen from "../components/QuestionScreen";
import StartScreen from "../components/StartScreen";
import { getTemplates, saveVoting } from "../utils/apiClient";

const DEFAULT_VOTING = { questions: [] };

let voting = DEFAULT_VOTING;

const Home = () => {
  const [templates, setTemplates] = useState([]);
  const [questionId, setQuestionId] = useState(-1);
  const [isError, setIsError] = useState(false);

  const loadTemplate = (name) => {
    const filteredTemplates = templates.filter(
      (template) => template.name === name
    );
    if (filteredTemplates.length > 0) {
      voting = filteredTemplates[0];
      setQuestionId(0);
    } else {
      voting = DEFAULT_VOTING;
    }
    console.log("set voting:", voting);
  };
  const voteAnswer = (answer) => {
    if (answer) {
      voting.questions[questionId][answer].is_selected = "x";
    } else {
      voting.questions[questionId].a1.is_selected = "x";
      voting.questions[questionId].a2.is_selected = "x";
    }
    console.log("update voting:", voting);

    if (questionId === voting.questions.length - 1) {
      sendVoting();
    }

    setQuestionId(questionId + 1);
  };
  const skipQuestion = () => {
    if (questionId === voting.questions.length - 1) {
      sendVoting();
    }

    setQuestionId(questionId + 1);
  };
  const sendVoting = async () => {
    try {
      await saveVoting(voting);
      setIsError(false);
      console.log("saved voting");
    } catch (e) {
      setIsError(true);
      console.error(e);
    }
  };

  useEffect(async () => {
    setTemplates(await getTemplates());
  }, []);

  if (questionId === -1) {
    return (
      <StartScreen templates={templates} onSelectTemplate={loadTemplate} />
    );
  } else if (questionId < voting.questions.length) {
    return (
      <QuestionScreen
        questionPair={voting.questions[questionId]}
        curPosition={questionId + 1}
        maxPosition={voting.questions.length}
        onSelectAnswer={voteAnswer}
        onSkipQuestion={skipQuestion}
      />
    );
  }
  return (
    <FinalScreen
      isError={isError}
      datasetName={voting.name}
      onRetry={sendVoting}
    />
  );
};

export default Home;
