import React from "react";
import Typography from "@mui/material/Typography";
import Container from "@mui/material/Container";
import { styled } from "@mui/material/styles";
import Box from "@mui/material/Box";
import ButtonBase from "@mui/material/ButtonBase";
import Dialog from "@mui/material/Dialog";
import RecipePage from "./recipePage";
import Toolbar from "@mui/material/Toolbar";
import IconButton from "@mui/material/IconButton";
import RefreshIcon from "@mui/icons-material/Refresh";
import Alert from "@mui/material/Alert";
import Divider from "@mui/material/Divider";
const ImageButton = styled(ButtonBase)(({ theme }) => ({
  position: "relative",
  height: 200,
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

export default function ThemePage() {
  const [themes, setThemes] = React.useState([]);
  const [posts, setPosts] = React.useState([]);
  const [open, setOpen] = React.useState(false);
  const [recId, setRecID] = React.useState(0);
  const [alert, setAlert] = React.useState(
    window.localStorage.getItem("is_cold") === "true"
  );
  const handleOpen = () => {
    setOpen(true);
  };
  const handleClose = () => {
    setOpen(false);
  };
  const publicUrl = process.env.PUBLIC_URL;
  React.useEffect(() => {
    get_themes();
  }, []);

  const get_themes = () => {
    const requestOptions = {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
      },
    };

    fetch(`/api/v1/num_themes`, requestOptions)
      .then((response) => response.json())
      .then((json) => {
        setThemes(selectIndex(json["num"], 5));
        get_posts(themes).then((json) => {
          setPosts(json);
          console.log(posts);
        });
      })
      .catch((error) => {});
  };
  const get_posts = (theme_list) => {
    const requestOptions = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
      },
      body: JSON.stringify({
        themes: theme_list,
      }),
    };

    return fetch(`/api/v1/themes`, requestOptions)
      .then((response) => response.json())
      .then((json) => {
        return json["articles"];
      })
      .catch((error) => {});
  };

  const refresh_theme = (index, id) => {
    const requestOptions = {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
      },
    };

    fetch(`/api/v1/theme/${id}`, requestOptions)
      .then((response) => response.json())
      .then((json) => {
        let temp = [...posts];
        temp.splice(index, 1, json);
        setPosts(temp);
      })
      .catch((error) => {});
  };

  const selectIndex = (totalIndex, selectingNumber) => {
    let randomIndexArray = [];
    for (let i = 0; i < selectingNumber; i++) {
      //check if there is any duplicate index
      let randomNum = Math.floor(Math.random() * totalIndex);
      if (randomIndexArray.indexOf(randomNum) === -1) {
        randomIndexArray.push(randomNum);
      } else {
        //if the randomNum is already in the array retry
        i--;
      }
    }
    return randomIndexArray;
  };
  return (
    <div className="outer-div">
      {alert ? (
        <Alert
          severity="info"
          onClose={() => {
            setAlert(false);
          }}
        >
          맞춤 추천 기능은 레시피 평점 이력이 쌓이면 사용 가능합니다!
        </Alert>
      ) : (
        ""
      )}

      <Container component="main" maxWidth="xl">
        <Typography variant="h3">테마별 레시피 추천</Typography>
        <span>&nbsp;&nbsp;&nbsp;</span>
        <Divider />
        <span>&nbsp;&nbsp;&nbsp;</span>
        <Box textAlign="center">
          <IconButton
            onClick={() => {
              get_themes();
            }}
          >
            Refresh
            <RefreshIcon />
          </IconButton>
        </Box>

        {posts.map((item, index) => (
          <div>
            <Toolbar>
              <Typography variant="h4" key={index}>
                {`#${item.theme_title}`}
              </Typography>
              <IconButton
                onClick={() => {
                  refresh_theme(index, item.theme_id);
                }}
              >
                <RefreshIcon />
              </IconButton>
            </Toolbar>

            <Box
              sx={{
                display: "flex",
                flexWrap: "wrap",
                minWidth: 300,
                width: "100%",
              }}
            >
              {item.samples.map((recipe) => (
                <ImageButton
                  style={{ width: "20%" }}
                  onClick={() => {
                    setRecID(recipe.id);
                    handleOpen();
                  }}
                >
                  <ImageSrc
                    style={{
                      backgroundImage: `url(${
                        recipe.image === "0.0"
                          ? `${publicUrl}/images/logo_g.png`
                          : recipe.image
                      })`,
                    }}
                  />
                  <ImageBackdrop className="MuiImageBackdrop-root" />
                  <Image>
                    <Typography
                      component="span"
                      variant="subtitle1"
                      color="inherit"
                      sx={{
                        position: "relative",
                        p: 4,
                        pt: 2,
                        pb: (theme) => `calc(${theme.spacing(1)} + 6px)`,
                      }}
                    >
                      {recipe.title}
                    </Typography>

                    <ImageMarked className="MuiImageMarked-root" />
                  </Image>
                </ImageButton>
              ))}
            </Box>
            <span>&nbsp;&nbsp;&nbsp;</span>
          </div>
        ))}
        <Dialog
          open={open}
          onClose={handleClose}
          aria-labelledby="scroll-dialog-title"
          aria-describedby="scroll-dialog-description"
        >
          <RecipePage id={recId} />
        </Dialog>
      </Container>
    </div>
  );
}
