import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import SignIn from "./pages/signin";
import SignUp from "./pages/signup";
import MainPage from "./pages/mainPage";
import MyPage from "./pages/myPage";
import reportWebVitals from "./reportWebVitals";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { createTheme, ThemeProvider } from "@mui/material/styles";

function Page() {
  const userid = localStorage.getItem("userid");
  if (!userid) {
    return (
      <BrowserRouter>
        <Routes>
          <Route path={"/*"} element={<Navigate replace to={"/signin"} />} />
          <Route path={"/signin"} element={<SignIn />} />
          <Route path={"/signup"} element={<SignUp />} />
        </Routes>
      </BrowserRouter>
    );
  } else {
    return (
      <BrowserRouter>
        <Routes>
          <Route path={"/*"} element={<Navigate replace to={"/main"} />} />
          <Route path={"/main"} element={<MainPage />} />
          <Route path={"/mypage"} element={<MyPage />} />
        </Routes>
      </BrowserRouter>
    );
  }
}

const theme = createTheme();
const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <Page />
    </ThemeProvider>
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
