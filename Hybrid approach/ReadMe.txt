full set of rules that were derived during training:
|--- drama <= 0.50
|   |--- sci <= 1.50
|   |   |--- animation <= 0.50
|   |   |   |--- class: 1
|   |   |--- animation >  0.50
|   |   |   |--- class: 1
|   |--- sci >  1.50
|   |   |--- dialogue <= 0.50
|   |   |   |--- class: 1
|   |   |--- dialogue >  0.50
|   |   |   |--- class: 1
|--- drama >  0.50
|   |--- provoking <= 0.50
|   |   |--- war <= 0.50
|   |   |   |--- class: 1
|   |   |--- war >  0.50
|   |   |   |--- class: 1
|   |--- provoking >  0.50
|   |   |--- swank <= 0.50
|   |   |   |--- class: 1
|   |   |--- swank >  0.50
|   |   |   |--- class: 0

heading : Explanation of the Rules
Each rule is a condition that splits the data based on the content features of the movies. The key features (e.g., drama, sci, animation) come from the tags and genres used in the vectorizer.

heading : Key Features:
drama: Indicates whether the movie is classified under the "Drama" genre.
sci: Related to science fiction or similar content.
animation: Indicates the presence of animation content.
dialogue: Refers to user-generated tags indicating emphasis on dialogue-driven movies.
provoking: Indicates whether the movie has themes considered thought-provoking.
war: Refers to war-related themes or genres.
swank: Likely a user-generated tag describing a specific movie style or theme.


heading: Decision Path for Fantastic Mr. Fox

Rule 1: |--- drama <= 0.50
Rule 2: |   |--- sci <= 1.50
Rule 3: |   |   |--- animation <= 0.50
Rule 5: |   |   |--- animation >  0.50



