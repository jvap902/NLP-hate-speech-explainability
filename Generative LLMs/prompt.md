Você é um especialista em Processamento de Linguagem Natural e Explainable AI (XAI).

Sua tarefa é analisar o arquivo "instances.json" fornecido e identificar quais termos/palavras-chave do texto de cada instância contribuem para as classes de discurso de ódio atribuídas a ela. 

Instruções da Tarefa:
1. Leia cada objeto contido no arquivo "instances.json".
2. Para cada classe de ódio atribuída a uma instância, identifique no texto original quais palavras ou expressões específicas fundamentam essa classificação.
3. Extraia APENAS palavras que estejam literalmente presentes no texto da instância. Não parafraseie ou adicione termos novos.
4. Mantenha estritamente o "id" original de cada instância presente no JSON. Se uma mesma instância tiver mais de uma classe atribuída, crie uma linha no CSV para cada par (id, class).

Formato de Saída:
Sua resposta deve conter EXCLUSIVAMENTE o conteúdo do arquivo CSV, exatamente no mesmo formato e estrutura de colunas do arquivo "example.csv":

id,class,relevant_words

Regras de Formatação do CSV:
- Coluna 1 (id): O ID exato da instância no JSON.
- Coluna 2 (class): O nome do tipo/classe de discurso de ódio analisado.
- Coluna 3 (relevant_words): Uma representação em array JSON das palavras extraídas. Como se trata de um arquivo CSV, aplique estritamente o escape padrão de aspas duplas (duplicando as aspas internas e envolvendo o campo por aspas duplas externas).

Exemplo exato de linha no CSV:

5780,aporophobia,"[""pobres"", ""lascar""]"

Responda APENAS com o código do CSV dentro de um bloco de código Markdown (csv ... ).
Não inclua NENHUM texto introdutório, explicações, saudações ou comentários antes ou depois do CSV.