import * as React from "react";
import Button from "@mui/material/Button";
import CssBaseline from "@mui/material/CssBaseline";
import TextField from "@mui/material/TextField";
import Link from "@mui/material/Link";
import Grid from "@mui/material/Grid";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Container from "@mui/material/Container";
import Backdrop from "@mui/material/Backdrop";
import CircularProgress from "@mui/material/CircularProgress";
import swal from "sweetalert";

export default function SignUp() {
  let [name, setName] = React.useState("");
  let [pwd, setPwd] = React.useState("");
  let [pwdCheck, setPwdCheck] = React.useState("");
  let [loading, setLoading] = React.useState(false);
  const validate = (response) => {
    if (response["state"] === "Approved") {
      swal("Approved", "Welcome!", "success", {
        buttons: false,
        timer: 2000,
      }).then((value) => {
        window.location.href = "/signin";
      });
    } else if (response["detail"] === "duplicate error") {
      swal("Denied", "이미 존재하는 name 입니다!", "error");
    } else {
      swal("Denied", "name 혹은 password의 형식이 잘못되었습니다!", "error");
    }
  };
  const signup = (userData) => {
    setLoading(true);
    const requestOptions = {
      method: "POST",
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(userData),
    };
    fetch("/api/v1/signup", requestOptions)
      .then((response) => response.json())
      .then((json) => {
        setLoading(false);
        validate(json);
      });
  };
  const handleClick = (event) => {
    signup({
      name: name,
      password: pwd,
    });
  };
  return (
    <Container component="main" maxWidth="xs">
      <CssBaseline />
      <Box
        sx={{
          marginTop: 8,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
        }}
      >
        <Typography component="h1" variant="h5">
          Sign up
        </Typography>
        <Box component="form" noValidate sx={{ mt: 3 }}>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                required
                fullWidth
                label="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
              <Typography fontSize={10}>
                4자 이상의 소문자, 숫자만 사용해 주세요
              </Typography>
            </Grid>
            <Grid item xs={12}>
              <TextField
                required
                fullWidth
                label="Password"
                type="password"
                onChange={(e) => setPwd(e.target.value)}
              />
              <Typography fontSize={10}>
                4자 이상의 소문자, 숫자만 사용해 주세요
              </Typography>
            </Grid>
            <Grid item xs={12}>
              <TextField
                required
                fullWidth
                label="PasswordCheck"
                type="password"
                onChange={(e) => setPwdCheck(e.target.value)}
              />
            </Grid>
            <Grid item xs={12}>
              {pwd === pwdCheck ? (
                <div />
              ) : (
                <Typography color="red" fontSize={12}>
                  비밀번호가 일치하지않습니다.
                </Typography>
              )}
            </Grid>
          </Grid>
          {pwd === pwdCheck && name.length >= 4 && pwd.length >= 4 ? (
            <Button
              fullWidth
              variant="contained"
              sx={{ mt: 3, mb: 2 }}
              onClick={handleClick}
            >
              Sign Up
            </Button>
          ) : (
            <Button
              disabled
              fullWidth
              variant="contained"
              sx={{ mt: 3, mb: 2 }}
            >
              Sign Up
            </Button>
          )}
          <Grid container justifyContent="flex-end">
            <Grid item>
              <Link href="/signin" variant="body2">
                이미 계정이 있다면 로그인하세요!
              </Link>
            </Grid>
          </Grid>
        </Box>
      </Box>
      <Backdrop open={loading}>
        <CircularProgress color="inherit" />
      </Backdrop>
    </Container>
  );
}
