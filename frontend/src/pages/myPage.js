import React from "react";
import Button from "@mui/material/Button";
import AppBar from "@mui/material/AppBar";
import Box from "@mui/material/Box";
import Toolbar from "@mui/material/Toolbar";
import IconButton from "@mui/material/IconButton";
import AccountCircle from "@mui/icons-material/AccountCircle";
import Container from "@mui/material/Container";
import Divider from "@mui/material/Divider";
import "./hanggi.css";
import Tooltip from "@mui/material/Tooltip";
import Logout from "@mui/icons-material/Logout";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import ListItemIcon from "@mui/material/ListItemIcon";
import HomeIcon from "@mui/icons-material/Home";
import ArrowBackIosIcon from "@mui/icons-material/ArrowBackIos";
import { Typography } from "@mui/material";
import MainCard from "./mainCard";
import { DataGrid } from "@mui/x-data-grid";
import Dialog from "@mui/material/Dialog";
import RecipePage from "./recipePage";

export default function MyPage() {
  const [dopen, setDOpen] = React.useState(false);
  const [recId, setRecID] = React.useState(0);
  const [anchorEl, setAnchorEl] = React.useState(null);
  const columns = [
    { field: "id", headerName: "" },
    { field: "recipe_id", headerName: "ID", width: 200 },
    { field: "score", headerName: "평점", width: 200 },
    { field: "date", headerName: "날짜", width: 300 },
  ];
  const rows = JSON.parse(window.localStorage.getItem("interactions"));
  const open = Boolean(anchorEl);
  const logout = () => {
    window.localStorage.clear();
    window.location.href = "/";
  };
  const publicUrl = process.env.PUBLIC_URL;
  const handleDOpen = () => {
    setDOpen(true);
  };
  const handleDClose = () => {
    setDOpen(false);
  };
  const handleClick = (event) => {
    setAnchorEl(event.currentTarget);
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
              <MenuItem>
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
      &nbsp;&nbsp;
      <IconButton
        onClick={() => {
          window.location.href = "/main";
        }}
      >
        <ArrowBackIosIcon />
        <Typography>돌아가기</Typography>
      </IconButton>
      <div className="outer-div">
        <Container component="main" maxWidth="xl">
          <MainCard title={"마이페이지"}>
            <Typography variant="subtitle1">나의 추천 목록</Typography>
            <Divider />
            <span>&nbsp;&nbsp;&nbsp;</span>
            <div style={{ height: 650, width: "100%" }}>
              <DataGrid
                rows={rows}
                columns={columns}
                pageSize={10}
                rowsPerPageOptions={[10]}
                onRowClick={(v) => {
                  setRecID(v.row.recipe_id);
                  handleDOpen();
                }}
              />
            </div>
          </MainCard>
        </Container>
        <Dialog
          open={dopen}
          onClose={handleDClose}
          aria-labelledby="scroll-dialog-title"
          aria-describedby="scroll-dialog-description"
        >
          <RecipePage id={recId} />
        </Dialog>
      </div>
    </div>
  );
}
