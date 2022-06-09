import React from "react";
import Button from "@mui/material/Button";
import AppBar from "@mui/material/AppBar";
import Box from "@mui/material/Box";
import Toolbar from "@mui/material/Toolbar";
import IconButton from "@mui/material/IconButton";
import AccountCircle from "@mui/icons-material/AccountCircle";
import Container from "@mui/material/Container";
import "./hanggi.css";
import Tooltip from "@mui/material/Tooltip";
import Logout from "@mui/icons-material/Logout";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import ListItemIcon from "@mui/material/ListItemIcon";
import HomeIcon from "@mui/icons-material/Home";
import Switch from "@mui/material/Switch";
export default function TopBar(props) {
  const [anchorEl, setAnchorEl] = React.useState(null);
  const [cold, setCold] = React.useState(
    window.localStorage.getItem("is_cold") === "true" ? true : false
  );
  const open = Boolean(anchorEl);
  const logout = () => {
    window.localStorage.clear();
    window.location.href = "/";
  };
  const publicUrl = process.env.PUBLIC_URL;
  const handleSwitch = () => {
    console.log(cold);
    props.setSwitch(!props.switch);
  };
  const handleClick = (event) => {
    setAnchorEl(event.currentTarget);
    console.log(props.log);
  };
  const handleClose = () => {
    setAnchorEl(null);
  };
  return (
    <div>
      <AppBar position="static" color="inherit">
        <Container maxWidth="xl">
          <Toolbar disableGutters varient="dense">
            <Button>
              <img
                src={`${publicUrl}/images/logo_m.png`}
                alt="smalllogo"
                className="smalllogo"
              />
            </Button>
            <Box
              sx={{ flexGrow: 1, display: { xs: "none", md: "flex" } }}
            ></Box>
            {cold === true ? (
              <>
                <Switch disabled />
              </>
            ) : (
              <Switch
                checked={props.switch}
                onChange={handleSwitch}
                inputProps={{ "aria-label": "controlled" }}
              />
            )}

            <Box sx={{ flexGrow: 0 }}>
              <IconButton
                size="large"
                aria-label="show more"
                // aria-controls={mobileMenuId}
                // aria-haspopup="true"
                // onClick={handleMobileMenuOpen}
                onClick={handleClick}
                color="inherit"
                sx={{ p: 0 }}
              >
                <Tooltip title={window.localStorage.getItem("name")}>
                  <AccountCircle />
                </Tooltip>
              </IconButton>
            </Box>
            <Menu
              anchorEl={anchorEl}
              id="account-menu"
              open={open}
              onClose={handleClose}
              onClick={handleClose}
              PaperProps={{
                elevation: 0,
                sx: {
                  overflow: "visible",
                  filter: "drop-shadow(0px 2px 8px rgba(0,0,0,0.32))",
                  mt: 1.5,
                  "& .MuiAvatar-root": {
                    width: 32,
                    height: 32,
                    ml: -0.5,
                    mr: 1,
                  },
                  "&:before": {
                    content: '""',
                    display: "block",
                    position: "absolute",
                    top: 0,
                    right: 14,
                    width: 10,
                    height: 10,
                    bgcolor: "background.paper",
                    transform: "translateY(-50%) rotate(45deg)",
                    zIndex: 0,
                  },
                },
              }}
              transformOrigin={{ horizontal: "right", vertical: "top" }}
              anchorOrigin={{ horizontal: "right", vertical: "bottom" }}
            >
              <MenuItem
                onClick={() => {
                  window.location.href = "/mypage";
                }}
              >
                <ListItemIcon>
                  <HomeIcon fontSize="small" />
                </ListItemIcon>
                MyPage
              </MenuItem>
              <MenuItem onClick={logout}>
                <ListItemIcon>
                  <Logout fontSize="small" />
                </ListItemIcon>
                Logout
              </MenuItem>
            </Menu>
          </Toolbar>
        </Container>
      </AppBar>
    </div>
  );
}
