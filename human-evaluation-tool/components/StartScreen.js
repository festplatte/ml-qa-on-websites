import React from "react";
import Button from "./Button";
import Title from "./Title";

const StartScreen = ({ templates, onSelectTemplate }) => (
  <div className="py-12 bg-white">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <Title title="Welcome">
        Please select a question collection that you would like to rate
      </Title>
      <div className="mt-10">
        <dl className="space-y-10 md:space-y-0 md:grid md:grid-cols-2 md:gap-x-8 md:gap-y-10">
          <div className="flex">
            <div className="ml-4">
              <dt className="text-lg leading-6 font-medium text-gray-900">
                Introduction
              </dt>
              <dd className="mt-2 text-base text-gray-500">
                Thank you for participating in this human evaluation. We've
                built a question answering system that should be able to answer
                questions by reading the content of a website. The question
                collections on the right side contain questions from the website
                of the Canadian city of Reddeer that could be asked by
                inhabitants about the city organization.
                <br />
                <br />
                The question collections contain two answers for each question,
                some written by humans, some generated by a machine. We want you
                to select the answer you think is better suited as a response to
                the corresponding question in order to find out how good the
                generated answers are compared to the human answers. You can
                also decide that both answers are equally good.
                <br />
                <br />
                Your rating will be submitted anonymously after you've rated the
                last question-answer-pair. Since we want to make sure that each
                question collection is rated as often as the other ones, please
                have a look at this survey{" "}
                <a
                  href="https://terminplaner4.dfn.de/mEt13z5GPfkDY5nF"
                  target="_blank"
                >
                  https://terminplaner4.dfn.de/mEt13z5GPfkDY5nF
                </a>
                , choose the one with the fewest ratings and enter your
                selection.
              </dd>
            </div>
          </div>
          <div className="flex">
            <div className="ml-4">
              <dt className="text-lg leading-6 font-medium text-gray-900">
                Question collections
              </dt>
              <dd className="mt-2 text-base text-gray-500">
                {" "}
                {templates &&
                  templates.map(({ name }) => (
                    <Button onClick={() => onSelectTemplate(name)}>
                      {name}
                    </Button>
                  ))}
              </dd>
            </div>
          </div>
        </dl>
      </div>
      <div className="mt-10"></div>
    </div>
  </div>
);

export default StartScreen;