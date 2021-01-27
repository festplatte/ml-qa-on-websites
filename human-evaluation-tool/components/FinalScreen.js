import React from "react";
import Button from "./Button";
import Title from "./Title";

const FinalScreen = ({ isError, datasetName, onRetry }) => (
  <div className="py-12 bg-white">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <Title title="Finished">
        You have rated all answers of the selected dataset <b>{datasetName}</b>.
        Thank you for participating!
      </Title>
      <div className="mt-10">
        <dl className="space-y-10 md:space-y-0 md:grid md:grid-cols-2 md:gap-x-8 md:gap-y-10">
          {isError ? (
            <div className="flex">
              <div className="ml-4">
                <dt className="text-lg leading-6 font-medium text-gray-900">
                  Error
                </dt>
                <dd className="mt-2 text-base text-gray-500">
                  An error occured while trying to save your ratings. Please
                  retry.
                </dd>
                <Button onClick={onRetry}>Retry</Button>
              </div>
            </div>
          ) : (
            <div className="flex">
              <div className="ml-4">
                <dt className="text-lg leading-6 font-medium text-gray-900">
                  Success
                </dt>
                <dd className="mt-2 text-base text-gray-500">
                  Your ratings for the question collection <b>{datasetName}</b>{" "}
                  have been saved. Please don't forget to let us know which
                  collection you've rated on{" "}
                  <a
                    href="https://terminplaner4.dfn.de/mEt13z5GPfkDY5nF"
                    target="_blank"
                  >
                    https://terminplaner4.dfn.de/mEt13z5GPfkDY5nF
                  </a>
                  .
                </dd>
              </div>
            </div>
          )}
        </dl>
      </div>
    </div>
  </div>
);

export default FinalScreen;
