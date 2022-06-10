import React from "react";
import ing_list from "./ingredients.json";
import Box from "@mui/material/Box";
import Toolbar from "@mui/material/Toolbar";
import RecipePage from "./recipePage";
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
import Slider from "@mui/material/Slider";
import Checkbox from "@mui/material/Checkbox";
import Divider from "@mui/material/Divider";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import Backdrop from "@mui/material/Backdrop";
import CircularProgress from "@mui/material/CircularProgress";
import "./hanggi.css";
import MobileStepper from "@mui/material/MobileStepper";
import KeyboardArrowLeft from "@mui/icons-material/KeyboardArrowLeft";
import KeyboardArrowRight from "@mui/icons-material/KeyboardArrowRight";
import SwipeableViews from "react-swipeable-views";
import { autoPlay } from "react-swipeable-views-utils";
import { useTheme, styled } from "@mui/material/styles";
import Dialog from "@mui/material/Dialog";
import ButtonBase from "@mui/material/ButtonBase";
const ImageButton = styled(ButtonBase)(({ theme }) => ({
  position: "relative",
  height: 500,
  [theme.breakpoints.down("sm")]: {
    width: "100% !important", // Overrides inline-style
    height: 100,
  },
  "&:hover, &.Mui-focusVisible": {
    zIndex: 1,
    "& .MuiImageBackdrop-root": {
      opacity: 0.15,
    },
    "& .MuiImageMarked-root": {
      opacity: 0,
    },
    "& .MuiTypography-root": {
      border: "4px solid currentColor",
    },
  },
}));

const ImageSrc = styled("span")({
  position: "absolute",
  left: 0,
  right: 0,
  top: 0,
  bottom: 0,
  backgroundSize: "cover",
  backgroundPosition: "center 40%",
});

const Image = styled("span")(({ theme }) => ({
  position: "absolute",
  left: 0,
  right: 0,
  top: 0,
  bottom: 0,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  color: theme.palette.common.white,
}));

const ImageBackdrop = styled("span")(({ theme }) => ({
  position: "absolute",
  left: 0,
  right: 0,
  top: 0,
  bottom: 0,
  backgroundColor: theme.palette.common.black,
  opacity: 0.4,
  transition: theme.transitions.create("opacity"),
}));

const ImageMarked = styled("span")(({ theme }) => ({
  height: 3,
  width: 18,
  backgroundColor: theme.palette.common.white,
  position: "absolute",
  bottom: -2,
  left: "calc(50% - 9px)",
  transition: theme.transitions.create("opacity"),
}));
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
  const [recId, setRecID] = React.useState(0);
  const [open, setOpen] = React.useState(false);
  const [start, setStart] = React.useState(false);
  const [maxSteps, setMaxSteps] = React.useState(10);
  const handleOpen = () => {
    setOpen(true);
  };
  const handleClose = () => {
    setOpen(false);
  };
  const theme = useTheme();

  const [activeStep, setActiveStep] = React.useState(0);
  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleStepChange = (step) => {
    setActiveStep(step);
  };
  const AutoPlaySwipeableViews = autoPlay(SwipeableViews);

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
        setMaxSteps(json["lists"].length);
        setStart(true);
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
        <Typography variant="h4">맞춤형 레시피 추천</Typography>
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
        {error ? (
          <Container maxWidth="xl">
            <img
              src={`${publicUrl}/images/prog_w.png`}
              className="recipe-image"
            />
          </Container>
        ) : start === true ? (
          recipes.length === 0 ? (
            <Container maxWidth="xl">
              <img
                src={`${publicUrl}/images/none_w.png`}
                className="recipe-image"
              />
            </Container>
          ) : (
            <Container maxWidth="xl">
              <AutoPlaySwipeableViews
                axis={theme.direction === "rtl" ? "x-reverse" : "x"}
                index={activeStep}
                onChangeIndex={handleStepChange}
                enableMouseEvents
              >
                {recipes.map((recipe, index) => (
                  <div key={index}>
                    {Math.abs(activeStep - index) <= 2 ? (
                      <ImageButton
                        style={{ width: "100%" }}
                        onClick={() => {
                          setRecID(recipe.id);
                          handleOpen();
                        }}
                      >
                        <ImageSrc
                          style={{
                            backgroundImage: `url(${
                              recipe.url === "0.0"
                                ? `${publicUrl}/images/logo_g.png`
                                : recipe.url
                            })`,
                          }}
                        />
                        <ImageBackdrop className="MuiImageBackdrop-root" />
                        <Image>
                          <Typography
                            component="span"
                            variant="h6"
                            color="inherit"
                            sx={{
                              position: "relative",
                              p: 4,
                              pt: 2,
                              pb: (theme) => `calc(${theme.spacing(1)} + 6px)`,
                            }}
                          >
                            {recipe.name}
                          </Typography>
                          <ImageMarked className="MuiImageMarked-root" />
                        </Image>
                      </ImageButton>
                    ) : null}
                  </div>
                ))}
              </AutoPlaySwipeableViews>
              <MobileStepper
                steps={maxSteps}
                position="static"
                activeStep={activeStep}
                nextButton={
                  <Button
                    size="small"
                    onClick={handleNext}
                    disabled={activeStep === maxSteps - 1}
                  >
                    Next
                    {theme.direction === "rtl" ? (
                      <KeyboardArrowLeft />
                    ) : (
                      <KeyboardArrowRight />
                    )}
                  </Button>
                }
                backButton={
                  <Button
                    size="small"
                    onClick={handleBack}
                    disabled={activeStep === 0}
                  >
                    {theme.direction === "rtl" ? (
                      <KeyboardArrowRight />
                    ) : (
                      <KeyboardArrowLeft />
                    )}
                    Back
                  </Button>
                }
              />
            </Container>
          )
        ) : (
          ""
        )}

        <Dialog
          open={open}
          onClose={handleClose}
          aria-labelledby="scroll-dialog-title"
          aria-describedby="scroll-dialog-description"
        >
          <RecipePage id={recId} />
        </Dialog>
      </Container>
      <Backdrop open={loading}>
        <CircularProgress color="inherit" />
      </Backdrop>
    </div>
  );
}
