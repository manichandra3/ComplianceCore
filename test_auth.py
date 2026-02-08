from neo4j import GraphDatabase

# Hardcoding credentials to rule out Environment Variable issues
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password"  # This matches your Docker -e NEO4J_AUTH=neo4j/password

print(f"🔌 Attempting connection to {URI} as user '{USER}'...")

try:
    with GraphDatabase.driver(URI, auth=(USER, PASSWORD)) as driver:
        driver.verify_connectivity()
        print("✅ SUCCESS! Connected to Neo4j.")
        
        # Run a quick test query
        with driver.session() as session:
            result = session.run("RETURN 'Hello Graph' AS message")
            print(f"🎉 Database says: {result.single()['message']}")

except Exception as e:
    print(f"❌ CONNECTION FAILED: {e}")
    print("\nTroubleshooting Tips:")
    print("1. Did you change the password in the docker run command?")
    print("2. If you are on Linux with --network host, ensure no firewall blocks port 7687.")
