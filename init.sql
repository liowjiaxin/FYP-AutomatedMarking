CREATE DATABASE IF NOT EXISTS automarker;
USE automarker;

CREATE USER IF NOT EXISTS 'appuser'@'%' IDENTIFIED BY 'apppassword';
GRANT ALL PRIVILEGES ON automarker.* TO 'appuser'@'%';
FLUSH PRIVILEGES; 