from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
Walter Elias Disney (/ˈdɪzni/ DIZ-nee;[2] December 5, 1901 – December 15, 1966) was an American animator, film producer, voice actor, and entrepreneur. A pioneer of the American animation industry, he introduced several developments in the production of cartoons. As a film producer, he holds the record for most Academy Awards won (22) and nominations (59) by an individual. He was presented with two Golden Globe Special Achievement Awards and an Emmy Award, among other honors. Several of his films are included in the National Film Registry by the Library of Congress and have also been named as some of the best by the American Film Institute.

Born in Chicago in 1901 and raised largely in Missouri, Disney developed an early interest in drawing. He took art classes as a boy and took a job as a commercial illustrator at the age of 18. He moved to California in the early 1920s and set up the Disney Brothers Studio (now the Walt Disney Company) with his brother Roy. With Ub Iwerks, he developed the character Mickey Mouse in 1928, his first highly popular success; he also provided the voice for his creation in the early years. As the studio grew, he became more adventurous, introducing synchronized sound, full-color three-strip Technicolor, animated feature films and technical developments in cameras. The results, seen in features such as Snow White and the Seven Dwarfs (1937), Pinocchio, Fantasia (both 1940), Dumbo (1941), and Bambi (1942), furthered the development of animated film. New animated and live-action films followed after World War II, including Cinderella (1950), Sleeping Beauty (1959), and Mary Poppins (1964), the last of which received five Academy Awards.

In the 1950s, Disney expanded into the theme park industry, and in July 1955 he opened Disneyland in Anaheim, California. To fund the project he diversified into television programs, such as Walt Disney's Disneyland and The Mickey Mouse Club. He was also involved in planning the 1959 Moscow Fair, the 1960 Winter Olympics, and the 1964 New York World's Fair. In 1965, he began development of another theme park, Disney World, the heart of which was to be a new type of city, the "Experimental Prototype Community of Tomorrow" (EPCOT). Disney was a heavy smoker throughout his life and died of lung cancer in 1966 before either the park or the EPCOT project were completed.

Disney was a shy, self-deprecating and insecure man in private but adopted a warm and outgoing public persona. He had high standards and high expectations of those with whom he worked. Although there have been accusations that he was racist or antisemitic, they have been contradicted by many who knew him. The historiography of Disney has taken a variety of perspectives, ranging from views of him as a purveyor of homely patriotic values to being a representative of American cultural imperialism. Widely considered to be one of the most influential cultural figures of the 20th century, Disney remains an important presence in the history of animation and in the cultural history of the United States, where he is acknowledged as a national cultural icon. His film work continues to be shown and adapted, the Disney theme parks have grown in size and number around the world and his company has grown to become one of the world's largest mass media and entertainment conglomerates.
    """

    summary_template = """
given the information {information} about a person, I want you to create:
1. A short summary
2. Two interesting facts about the person
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(model="gpt-5", temperature=0)
    chain = summary_prompt_template | llm

    response = chain.invoke({"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
