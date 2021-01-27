import React from "react";
import AnswerBlock from "./AnswerBlock";
import Button from "./Button";
import Title from "./Title";

const QuestionScreen = ({
  questionPair,
  onSelectAnswer,
  onSkipQuestion,
  curPosition,
  maxPosition,
}) => (
  <div className="py-12 bg-white">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <Title
        title={questionPair.question}
        headline={`Human evaluation (${curPosition}/${maxPosition})`}
      >
        Please select the answer you think is better suited as a response to the
        question above.{" "}
        <button
          className="mr-5 bg-white p-2 border border-gray-300 rounded-md shadow-sm text-sm leading-4 font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          onClick={onSkipQuestion}
        >
          Skip
        </button>
      </Title>
      <div className="mt-10">
        <dl className="space-y-10 md:space-y-0 md:grid md:grid-cols-3 md:gap-x-8 md:gap-y-10">
          <AnswerBlock title="Answer 1" onClick={() => onSelectAnswer("a1")}>
            {questionPair.a1.answer}
          </AnswerBlock>
          <AnswerBlock title="Equal" onClick={() => onSelectAnswer()}>
            <i>Select this, if both answers are equally good.</i>
          </AnswerBlock>
          <AnswerBlock title="Answer 2" onClick={() => onSelectAnswer("a2")}>
            {questionPair.a2.answer}
          </AnswerBlock>
        </dl>
      </div>
    </div>
  </div>
);

export default QuestionScreen;
