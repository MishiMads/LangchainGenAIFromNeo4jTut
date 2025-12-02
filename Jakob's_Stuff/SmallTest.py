import datetime

student_name = "poo"
operation_name = "bæ"
is_correct = True
now_time = datetime.datetime.now()
if is_correct:
    query = f"""
        MATCH (s:Student {{_navn: '{student_name}'}}),
              (o:Operation {{name: '{operation_name}'}})
        MERGE (s)-[rel:QUESTIONS_ANSWERED]->(o)
        ON MATCH SET rel.correctAnswers = rel.correctAnswers + 1,
                      rel.totalAnswers = rel.totalAnswers + 1
        ON CREATE SET rel.correctAnswers = 1,
                      rel.incorrectAnswers = 0,
                      rel.totalAnswers = 1;
        """
else:
    query = f"""
        MATCH (s:Student {{_navn: '{student_name}'}}),
              (o:Operation {{name: '{operation_name}'}})
        MERGE (s)-[rel:QUESTIONS_ANSWERED]->(o)
        ON MATCH SET rel.incorrectAnswers = rel.incorrectAnswers + 1,
                      rel.totalAnswers = rel.totalAnswers + 1
        ON CREATE SET rel.correctAnswers = 0,
                      rel.incorrectAnswers = 1,
                      rel.totalAnswers = 1;
        """

query2 = f"""
    MERGE (s)-[rel2:LAST_ANSWERED]->(o)
    SET rel2.recency = {now_time}
    """

new_query = query + query2
print(new_query)