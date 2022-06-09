import React from "react";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Container from "@mui/material/Container";
import Stack from "@mui/material/Stack";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import MainCard from "./mainCard";
import Paper from "@mui/material/Paper";
import "./hanggi.css";
import { styled } from "@mui/material/styles";
import StickyNote2OutlinedIcon from "@mui/icons-material/StickyNote2Outlined";
import Toolbar from "@mui/material/Toolbar";
import Divider from "@mui/material/Divider";
import Rating from "@mui/material/Rating";
import Box from "@mui/material/Box";
const Item = styled(Paper)(({ theme }) => ({
  backgroundColor: theme.palette.mode === "dark" ? "#1A2027" : "#fff",
  ...theme.typography.body2,
  padding: theme.spacing(1),
  textAlign: "left",
  color: theme.palette.text.secondary,
}));

export default function RecipePage(props) {
  const [sended, setSended] = React.useState(false);
  const [rating, setRating] = React.useState(5);
  const [info, setInfo] = React.useState("load");
  const interactions = JSON.parse(window.localStorage.getItem("interactions"));
  const logs = JSON.parse(window.localStorage.getItem("log")).map((item) => {
    return parseInt(item, 10);
  });
  const publicUrl = process.env.PUBLIC_URL;
  const get_recipe = (id) => {
    const requestOptions = {
      method: "GET",
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
        "Content-Type": "application/json",
      },
    };
    fetch(`/api/v1/recipe/${id}`, requestOptions)
      .then((response) => response.json())
      .then((json) => {
        setInfo(json);
      });
  };
  const scoring = () => {
    const requestOptions = {
      method: "POST",
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_id: parseInt(window.localStorage.getItem("userid"), 10),
        recipe_id: info.id,
        rating: rating,
      }),
    };
    fetch("/api/v1/score", requestOptions)
      .then((response) => response.json())
      .then((json) => {
        for (let i = 0; i < json["interactions"].length; i++) {
          json["interactions"][i]["id"] = i;
        }
        window.localStorage.setItem(
          "interactions",
          JSON.stringify(json["interactions"])
        );
        window.localStorage.setItem("log", JSON.stringify(json["log"]));
        window.localStorage.setItem(
          "interaction_count",
          json["interaction_count"]
        );
        window.localStorage.setItem("is_cold", json["is_cold"]);
      });
  };
  React.useEffect(() => {
    get_recipe(props.id);
  }, []);
  return (
    <div className="outer-div">
      {info === "load" ? (
        <div />
      ) : (
        <Container component="main" maxWidth="xl">
          <MainCard title={info.name}>
            <img
              src={
                info.url === "0.0" ? `${publicUrl}/images/logo_g.png` : info.url
              }
              className="recipe-image"
              alt={info.id}
              loading="lazy"
            />
            <TableContainer comonenet={Paper}>
              <Table size="small" aria-label="a dense table">
                <TableHead>
                  <TableRow>
                    <TableCell align="right">
                      <Typography fontSize={12}>칼로리(kcal)</Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Typography fontSize={12}>탄수화물(g)</Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Typography fontSize={12}>단백질(g)</Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Typography fontSize={12}>지방(g)</Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Typography fontSize={12}>당류(g)</Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Typography fontSize={12}>나트륨(mg)</Typography>
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  <TableCell align="right">{info.calories}</TableCell>
                  <TableCell align="right">{info.carbohydrates}</TableCell>
                  <TableCell align="right">{info.protein}</TableCell>
                  <TableCell align="right">{info.totalfat}</TableCell>
                  <TableCell align="right">{info.sugar}</TableCell>
                  <TableCell align="right">{info.sodium}</TableCell>
                </TableBody>
              </Table>
            </TableContainer>
            <span>&nbsp;&nbsp;&nbsp;</span>
            <Divider />
            <Paper>
              <div className="outer-card">
                <Toolbar>
                  <StickyNote2OutlinedIcon />
                  <Typography fontSize={20} variant="h6">
                    조리 방법 (소요시간 : {info.minutes}분)
                  </Typography>
                </Toolbar>
                <Stack spacing={2}>
                  {[...info.steps].map((step, index) => (
                    <Item>{`${index + 1} :  ${step}`}</Item>
                  ))}
                </Stack>
                <span>&nbsp;&nbsp;&nbsp;</span>
                <Paper>
                  {logs.indexOf(info.id) === -1 ? (
                    <Toolbar>
                      <Rating
                        size="large"
                        name="half-rating"
                        value={rating}
                        onChange={(event, newValue) => {
                          setRating(newValue);
                        }}
                        disabled={sended}
                        precision={0.5}
                      />
                      &nbsp;
                      <Typography variant="h6">{rating}</Typography>
                      <Box
                        sx={{
                          flexGrow: 1,
                          display: { xs: "none", md: "flex" },
                        }}
                      ></Box>
                      <Button
                        disabled={sended}
                        onClick={() => {
                          scoring();
                          setSended(true);
                        }}
                      >
                        평가하기
                      </Button>
                    </Toolbar>
                  ) : (
                    <Toolbar>
                      <Rating
                        size="large"
                        value={interactions[logs.indexOf(info.id)]["score"]}
                        disabled
                      />
                      &nbsp;
                      <Typography variant="h6">
                        {interactions[logs.indexOf(info.id)]["score"]}
                      </Typography>
                      <Box
                        sx={{
                          flexGrow: 1,
                          display: { xs: "none", md: "flex" },
                        }}
                      ></Box>
                      <Button disabled>평가했음</Button>
                    </Toolbar>
                  )}
                </Paper>
                <span>&nbsp;&nbsp;&nbsp;</span>
              </div>
            </Paper>
          </MainCard>
        </Container>
      )}
    </div>
  );
}
