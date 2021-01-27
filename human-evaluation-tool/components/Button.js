import React from "react";

const Button = ({ children, ...props }) => (
  <button
    type="button"
    className="mt-5 mr-5 bg-white py-4 px-5 border border-gray-300 rounded-md shadow-sm text-md leading-4 font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
    {...props}
  >
    {children}
  </button>
);

export default Button;
