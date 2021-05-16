//
// Created by Khurram Javed on 2021-03-29.
//

#include "../../include/experiment/Database.h"
#include <string>
#include <iostream>
#include <mysql/mysql.h>
#include <cstring>

Database::Database() {
    this->mysql = mysql_init(NULL);

    if (!this->mysql) {
        puts("Init faild, out of memory?");
    }

    mysql_options(this->mysql, MYSQL_READ_DEFAULT_FILE, (void *) ".my.cnf");
}

/// Connets to the database and stores the connection in this->mysql
/// \return 0 if successfull, non-zero otherwise
int Database::connect() {

    this->mysql = mysql_init(NULL);

    if (!this->mysql) {
        puts("Init faild, out of memory?");
    }
//
    mysql_options(this->mysql, MYSQL_READ_DEFAULT_FILE, (void *) ".my.cnf");
    mysql_real_connect(this->mysql,       /* MYSQL structure to use */
                       NULL,  /* server hostname or IP address */
                       NULL,  /* mysql user */
                       NULL,   /* password */
                       NULL,        /* default database to use, NULL for none */
                       0,           /* port number, 0 for default */
                       NULL,        /* socket file or named pipe name */
                       CLIENT_FOUND_ROWS /* connection flags */ );
    return 0;

}

/// Connect to the database and execute USE this->db_name;
/// \return
int Database::connect_and_use(std::string database_name) {

    this->mysql = mysql_init(NULL);

    if (!this->mysql) {
        puts("Init faild, out of memory?");
    }
//
    mysql_options(this->mysql, MYSQL_READ_DEFAULT_FILE, (void *) ".my.cnf");
    mysql_real_connect(this->mysql,       /* MYSQL structure to use */
                       NULL,  /* server hostname or IP address */
                       NULL,  /* mysql user */
                       NULL,   /* password */
                       NULL,        /* default database to use, NULL for none */
                       0,           /* port number, 0 for default */
                       NULL,        /* socket file or named pipe name */
                       CLIENT_FOUND_ROWS /* connection flags */ );
    std::string use_query = "USE " + database_name + ";";
//    std::cout << "Running query " << use_query << std::endl;
    int selection = mysql_query(this->mysql, &use_query[0]);
    return 0;

}

int Database::create_database(const std::string &database_name) {
    std::string query = "CREATE DATABASE " + database_name + ";";
    std::cout << query << std::endl;
    this->connect();
    int val = mysql_query(this->mysql, &query[0]);
    if (val) {
        std::cout << "Database creation failed. Perhaps it already exists\n";
    }
    mysql_commit(this->mysql);
    mysql_close(this->mysql);
    return val;
}

int Database::run_query(std::string query, const std::string &database_name) {
    this->connect_and_use(database_name);
    mysql_query(this->mysql, &query[0]);
    return 1;
}


std::string Database::vec_to_tuple(std::vector<std::string> row, const std::string &padding) {
    std::string tup = "(";
    for (int counter = 0; counter < row.size() - 1; counter++) {
        tup += padding;
        tup += row[counter];
        tup += padding;
        tup += ",";
    }
    tup = tup + padding + row[row.size() - 1] + padding + " )";
    return tup;
}

int Database::add_rows_to_table(const std::string &database_name, const std::string &table,
                                const std::vector<std::string> &keys,
                                const std::vector<std::vector<std::string>> &values) {
    this->connect_and_use(database_name);
    for (auto &value : values) {
        std::string query = "INSERT INTO " + table + vec_to_tuple(keys, "") + " VALUES " + vec_to_tuple(value, "'");
        mysql_query(this->mysql, &query[0]);
    }
    mysql_commit(this->mysql);
    mysql_close(this->mysql);
    return 0;
}

int
Database::add_row_to_table(const std::string &database_name, const std::string &table, std::vector<std::string> keys,
                           std::vector<std::string> values) {
    std::string query = "INSERT INTO " + table + vec_to_tuple(keys, "") + " VALUES " + vec_to_tuple(values, "'");
    this->run_query(query, database_name);
    mysql_commit(this->mysql);
    mysql_close(this->mysql);
    return 0;
}

int Database::make_table(const std::string &database_name, const std::string &table, std::vector<std::string> keys,
                         std::vector<std::string> types,
                         std::vector<std::string> index_columns) {
    if (this->connect() == 0) {
        std::string query;
        query = "CREATE TABLE " + table + " (";
        if (keys.size() != types.size()) {
            std::cout << "SQL number of columns and number of types are not equal\n";
            exit(1);
        }
        for (int counter = 0; counter < keys.size(); counter++) {
            query += " " + keys[counter] + " " + types[counter] + ",";
        }
        query = query + " PRIMARY KEY(";
        for (int counter = 0; counter < index_columns.size() - 1; counter++) {
            query += " " + index_columns[counter] + " ,";
        }
        query = query + " " + index_columns[index_columns.size() - 1] + " ));";
        std::cout << "Creating table: " << query << std::endl;
        this->run_query(query, database_name);
    }
    return 1;

}

