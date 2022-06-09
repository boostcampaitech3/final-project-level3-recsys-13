import * as React from "react";
import Button from "@mui/material/Button";
import CssBaseline from "@mui/material/CssBaseline";
import TextField from "@mui/material/TextField";
import Link from "@mui/material/Link";
import Grid from "@mui/material/Grid";
import Box from "@mui/material/Box";
import Container from "@mui/material/Container";
import Backdrop from "@mui/material/Backdrop";
import CircularProgress from "@mui/material/CircularProgress";
import "./hanggi.css";
import swal from "sweetalert";
export default function SignIn(props) {
  let [name, setName] = React.useState("");
  let [pwd, setPwd] = React.useState("");
  let [loading, setLoading] = React.useState(false);
  const publicUrl = process.env.PUBLIC_URL;
  const validate = (response) => {
    if (response["state"] === "Approved") {
      swal("Approved", "Welcome!", "success", {
        buttons: false,
        timer: 2000,
      }).then((value) => {
        // console.log(response);
        window.localStorage.setItem("userid", response["user_id"]);
        window.localStorage.setItem("name", response["name"]);
        for (let i = 0; i < response["interactions"].length; i++) {
          response["interactions"][i]["id"] = i;
        }
        window.localStorage.setItem(
          "interactions",
          JSON.stringify(response["interactions"])
        );
        window.localStorage.setItem("log", JSON.stringify(response["log"]));
        window.localStorage.setItem(
          "interaction_count",
          response["interaction_count"]
        );
        window.localStorage.setItem("is_cold", response["is_cold"]);
        window.localStorage.setItem("pageSwitch", response["is_cold"]);
        window.location.href = "/main";
      });
    } else if (response["detail"] === "wrong password") {
      swal("Denied", "잘못된 비밀번호 입니다.", "error");
    } else {
      swal("Denied", "존재하지 않는 name 입니다.", "error");
    }
  };
  const login = (userData) => {
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
    fetch("/api/v1/signin", requestOptions)
      .then((response) => response.json())
      .then((json) => {
        setLoading(false);
        validate(json);
      });
  };
  const handleClick = (event) => {
    login({
      name: name,
      password: pwd,
    });
  };
  const keyHandler = (event) => {
    if (event.key === "Enter") {
      handleClick();
    }
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
        <img src={`${publicUrl}/images/logo.png`} alt="logo" className="logo" />
        {/* <Typography component="h1" variant="h5">
            Sign in
          </Typography> */}
        <Box component="form" noValidate sx={{ mt: 1 }}>
          <TextField
            margin="normal"
            required
            fullWidth
            id="names"
            label="Name"
            name="names"
            autoComplete="names"
            autoFocus
            onChange={(e) => {
              setName(e.target.value);
            }}
          />
          <TextField
            margin="normal"
            required
            fullWidth
            name="password"
            label="Password"
            type="password"
            id="password"
            autoComplete="current-password"
            onChange={(e) => {
              setPwd(e.target.value);
            }}
            onKeyDown={keyHandler}
          />
          {/* <FormControlLabel
              control={<Checkbox value="remember" color="primary" />}
              label="Remember me"
            /> */}
          <Button
            fullWidth
            variant="contained"
            sx={{ mt: 3, mb: 2 }}
            onClick={handleClick}
          >
            Sign In
          </Button>
          <Grid container>
            <Grid item xs></Grid>
            <Grid item>
              <Link href="/signup" variant="body2">
                {"아이디가 없다면 회원가입하세요!"}
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
