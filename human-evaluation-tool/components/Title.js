import React from "react";

const Title = ({ headline = "Human Evaluation", title, children }) => (
  <div className="lg:text-center">
    <h2 className="text-base text-indigo-600 font-semibold tracking-wide uppercase">
      {headline}
    </h2>
    <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
      {title}
    </p>
    <p className="mt-4 max-w-2xl text-xl text-gray-500 lg:mx-auto">
      {children}
    </p>
  </div>
);

export default Title;
