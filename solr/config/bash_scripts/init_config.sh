FILE=""
DIR="$1"
# init
# look for empty dira
if [ -d "$DIR" ]
then
	if [ "$(ls -A $DIR)" ]; then
        echo "$DIR is NOT Empty"
        echo "No Solr Config Initialization Required"
	else
    echo "$DIR is Empty"
    echo "---> Inititalizing Solr Config..."
    docker exec solr bin/solr zk upconfig -zkhost zoo:2181 -n tova_config -d /opt/solr/server/solr/configsets/tova_config
	fi
else
	echo "Directory $DIR not found."
    echo "---> Inititalizing Solr Config..."
    docker exec solr bin/solr zk upconfig -zkhost zoo:2181 -n tova_config -d /opt/solr/server/solr/configsets/tova_config
fi





