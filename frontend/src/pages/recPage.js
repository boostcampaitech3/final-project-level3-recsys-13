import React, { useState } from "react";
import ing_list from "./ingredients.json";
import Box from "@mui/material/Box";
import Toolbar from "@mui/material/Toolbar";
import IconButton from "@mui/material/IconButton";
import Chip from "@mui/material/Chip";
import Autocomplete from "@mui/material/Autocomplete";
import TextField from "@mui/material/TextField";
import Container from "@mui/material/Container";
import { matchSorter } from "match-sorter";
import FilterAltIcon from "@mui/icons-material/FilterAlt";
import Button from "@mui/material/Button";
import SendIcon from "@mui/icons-material/Send";
import Paper from "@mui/material/Paper";
import ToggleButton from "@mui/material/ToggleButton";
import Typography from "@mui/material/Typography";
import Grid from "@mui/material/Grid";
import Slider from "@mui/material/Slider";
import Checkbox from "@mui/material/Checkbox";
import Divider from "@mui/material/Divider";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import Backdrop from "@mui/material/Backdrop";
import CircularProgress from "@mui/material/CircularProgress";
import Carousel from "react-bootstrap/Carousel";
import "./hanggi.css";
export default function RecPage(props) {
  const publicUrl = process.env.PUBLIC_URL;
  const [ingList, setIngList] = React.useState(ing_list["ingredients"]);
  const [ingValue, setIngValue] = React.useState([]);
  const [selected, setSelected] = React.useState(false);
  const [minutes, setMinutes] = React.useState(30);
  const [minSwitch, setMinSwitch] = React.useState(false);
  const [sodium, setSodium] = React.useState(0);
  const [sodSwitch, setSodSwitch] = React.useState(false);
  const [sugar, setSugar] = React.useState(0);
  const [sugSwitch, setSugSwitch] = React.useState(false);
  const [carb, setCarb] = React.useState(0);
  const [protein, setProtein] = React.useState(0);
  const [fat, setFat] = React.useState(0);
  const [recipes, setRecipes] = React.useState([]);
  const [error, setError] = React.useState(false);
  const [loading, setLoading] = React.useState(false);
  const [index, setIndex] = useState(0);

  const handleSelect = (selectedIndex, e) => {
    setIndex(selectedIndex);
  };
  const handleCarb = (event, newAlignment) => {
    if (newAlignment !== null) {
      setCarb(newAlignment);
    }
  };
  const handleProtein = (event, newAlignment) => {
    if (newAlignment !== null) {
      setProtein(newAlignment);
    }
  };
  const handleFat = (event, newAlignment) => {
    if (newAlignment !== null) {
      setFat(newAlignment);
    }
  };
  const filterOptions = (options, { inputValue }) =>
    matchSorter(options, inputValue);
  const recommend = (userData) => {
    setError(false);
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
    fetch("/api/v1/recten", requestOptions)
      .then((response) => response.json())
      .then((json) => {
        setLoading(false);
        setRecipes(json["lists"]);
      })
      .catch((error) => {
        setError(true);
      });
  };
  const query = () => {
    recommend({
      user_id: Number(window.localStorage.getItem("userid")),
      on_off_button: [
        ingValue.length === 0 ? 0 : 1,
        sodSwitch ? 1 : 0,
        sugSwitch ? 1 : 0,
        minSwitch ? 1 : 0,
        carb,
        protein,
        fat,
      ],
      ingredients_ls: ingValue,
      max_sodium: sodium,
      max_sugar: sugar,
      max_minutes: minutes,
    });
  };
  return (
    <div className="outer-div">
      <Container component="main" maxWidth="xl">
        <Typography variant="h3">맞춤형 레시피 추천</Typography>
        <span>&nbsp;&nbsp;&nbsp;</span>
        <Divider />
        <span>&nbsp;&nbsp;&nbsp;</span>
        <Toolbar>
          <Autocomplete
            sx={{ width: "90%" }}
            multiple
            id="tags-outlined"
            size="small"
            options={ingList}
            getOptionLabel={(option) => option}
            filterSelectedOptions
            filterOptions={filterOptions}
            onChange={(e, value) => {
              setIngValue(value);
            }}
            renderInput={(params) => (
              <TextField
                {...params}
                label="재료"
                placeholder="가지고 있는 재료를 입력하세요!"
              />
            )}
          />
          <ToggleButton
            size="small"
            selected={selected}
            onChange={() => {
              setSelected(!selected);
            }}
          >
            <FilterAltIcon />
          </ToggleButton>
          <Box
            sx={{
              flexGrow: 1,
              display: { xs: "none", md: "flex" },
            }}
          ></Box>
          <Button variant="contained" onClick={query} endIcon={<SendIcon />} />
        </Toolbar>
        {selected ? (
          <Container maxWidth="xl">
            <Paper>
              <Toolbar>
                <Box sx={{ width: "12%" }}>
                  <Typography variant="subtitle2">조리시간(분)</Typography>
                </Box>
                <Box sx={{ width: "80%" }}>
                  <Slider
                    value={minutes}
                    onChange={(e) => {
                      setMinutes(e.target.value);
                    }}
                    valueLabelDisplay="auto"
                    disabled={!minSwitch}
                    step={10}
                    min={10}
                    max={300}
                  />
                </Box>
                <Box
                  sx={{
                    flexGrow: 1,
                    display: { xs: "none", md: "flex" },
                  }}
                />
                <Checkbox
                  checked={minSwitch}
                  onChange={() => {
                    setMinSwitch(!minSwitch);
                  }}
                  inputProps={{ "aria-label": "controlled" }}
                />
              </Toolbar>
              <Divider />
              <Toolbar>
                <Box sx={{ width: "12%" }}>
                  <Typography variant="subtitle2">나트륨(mg)</Typography>
                </Box>
                <Box sx={{ width: "80%" }}>
                  <Slider
                    value={sodium}
                    onChange={(e) => {
                      setSodium(e.target.value);
                    }}
                    valueLabelDisplay="auto"
                    disabled={!sodSwitch}
                    step={5}
                    min={10}
                    max={100}
                  />
                </Box>
                <Box
                  sx={{
                    flexGrow: 1,
                    display: { xs: "none", md: "flex" },
                  }}
                />
                <Checkbox
                  checked={sodSwitch}
                  onChange={() => {
                    setSodSwitch(!sodSwitch);
                  }}
                  inputProps={{ "aria-label": "controlled" }}
                />
              </Toolbar>
              <Divider />
              <Toolbar>
                <Box sx={{ width: "12%" }}>
                  <Typography variant="subtitle2">당류(g)</Typography>
                </Box>
                <Box sx={{ width: "80%" }}>
                  <Slider
                    value={sugar}
                    onChange={(e) => {
                      setSugar(e.target.value);
                    }}
                    valueLabelDisplay="auto"
                    disabled={!sugSwitch}
                    step={5}
                    min={5}
                    max={100}
                  />
                </Box>
                <Box
                  sx={{
                    flexGrow: 1,
                    display: { xs: "none", md: "flex" },
                  }}
                />
                <Checkbox
                  checked={sugSwitch}
                  onChange={() => {
                    setSugSwitch(!sugSwitch);
                  }}
                  inputProps={{ "aria-label": "controlled" }}
                />
              </Toolbar>
              <Toolbar>
                <Box sx={{ width: "30%" }}>
                  <Typography variant="subtitle2">탄단지조절</Typography>
                </Box>
                <Box sx={{ width: "30%" }}>
                  <ToggleButtonGroup
                    color="primary"
                    value={carb}
                    size="small"
                    exclusive
                    onChange={handleCarb}
                  >
                    <ToggleButton value={1}>저탄수</ToggleButton>
                    <ToggleButton value={0}>사용안함</ToggleButton>
                    <ToggleButton value={2}>고탄수</ToggleButton>
                  </ToggleButtonGroup>
                </Box>
                <Divider orientation="vertical" />
                <Box sx={{ width: "30%" }}>
                  <ToggleButtonGroup
                    color="primary"
                    value={protein}
                    size="small"
                    exclusive
                    onChange={handleProtein}
                  >
                    <ToggleButton value={1}>저단백</ToggleButton>
                    <ToggleButton value={0}>사용안함</ToggleButton>
                    <ToggleButton value={2}>고단백</ToggleButton>
                  </ToggleButtonGroup>
                </Box>
                <Divider orientation="vertical" />
                <Box sx={{ width: "30%" }}>
                  <ToggleButtonGroup
                    color="primary"
                    value={fat}
                    size="small"
                    exclusive
                    onChange={handleFat}
                  >
                    <ToggleButton value={1}>저지방</ToggleButton>
                    <ToggleButton value={0}>사용안함</ToggleButton>
                    <ToggleButton value={2}>고지방</ToggleButton>
                  </ToggleButtonGroup>
                </Box>
                <Divider orientation="vertical" />
              </Toolbar>
            </Paper>
          </Container>
        ) : (
          ""
        )}
        {recipes.length === 0 ? (
          <Typography>하이</Typography>
        ) : (
          <Carousel activeIndex={index} onSelect={handleSelect}>
            {recipes.map((recipe) => (
              <Carousel.Item>
                <img className="d-block w-100" src={`${recipe.url}`} />
                <Carousel.Caption>
                  <h3>{recipe.name}</h3>
                </Carousel.Caption>
              </Carousel.Item>
            ))}
          </Carousel>
        )}
      </Container>
      <Backdrop open={loading}>
        <CircularProgress color="inherit" />
      </Backdrop>
    </div>
  );
}
