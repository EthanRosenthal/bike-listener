# Run like
# nohup bash dump_db.sh pwd dat_dump.csv.gz
export PGPASSWORD=$1
psql -U postgres -w -h localhost -d stations -c "COPY status TO stdout DELIMITER '|' CSV HEADER" | gzip > $2
