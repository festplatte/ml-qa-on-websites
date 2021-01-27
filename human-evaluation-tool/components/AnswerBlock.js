import React from "react";
import Button from "./Button";

const AnswerBlock = ({ title, children, onClick }) => (
  <div className="flex">
    <div className="ml-4">
      <dt className="text-lg leading-6 font-medium text-gray-900">{title}</dt>
      <dd className="mt-2 text-base text-gray-500">{children}</dd>
      <Button onClick={onClick}>Select</Button>
    </div>
  </div>
);

export default AnswerBlock;
