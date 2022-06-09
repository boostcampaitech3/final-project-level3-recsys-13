import React from "react";
import TopBar from "./TopBar";
import ThemePage from "./themePage";
import RecPage from "./recPage";
export default function MainPage(props) {
  const [pageSwitch, setPageSwitch] = React.useState(
    !window.localStorage.getItem("pageSwitch")
  );
  return (
    <div>
      <TopBar switch={pageSwitch} setSwitch={setPageSwitch} />
      {pageSwitch ? <RecPage /> : <ThemePage />}
    </div>
  );
}
