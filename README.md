# Conditional-Text-Generation
### Overview

 - Fine tuning Huggingface GPT2 model to generate text.
 - Given a title and list of keywords, the model should be able to generate text
 - This task is reverse of Text Summarization

### Dataset
 - We have used random 40k samles of https://archive.ics.uci.edu/ml/datasets/News+Aggregator dataset to train the model
 - We extracted keywords from the different documents using https://github.com/ivanlai/Conditional_Text_Generation/blob/master/Keyword_extract.py depending on the TFIDF scores of the words

### Model
 - Since our task is to generate text based on Title and keywords we have provided training data in following format
```input = "<|BOS|>" + self.title[i] + "<|SEP|>" + self.keywords[i] + "<|SEP|>" + self.text[i] + "<|EOS|>"```
 - Unfreezed last 10 layers for training
 - Trained the model in GCP AI Platform with NVDIA Tesla T4 GPU
 - Trained for 6 epochs on the whole data. Following result was achieved
 
| Epoch	| Training Loss	| Validation Loss | 
| ------ |------ |------|
|0|	No log|	1.497857|
|1	|2.941100|	1.457414|
|2	|1.472000|	1.439115|
|3|	1.387700	|1.428377|
|4	|1.323000|	1.416111|
|5|	1.268000|	1.419795|

### Result
 - We need to provide Title and Keywords to get result
```
title = "We got a lot of grief when our photo became a meme"
keywords = ['train', 'lads', 'drinking', 'picture', 'funny', 'instagram']
kw = myDataset.join_keywords(keywords, randomize=False)

prompt = SPECIAL_TOKENS['bos_token'] + title + \
         SPECIAL_TOKENS['sep_token'] + kw + SPECIAL_TOKENS['sep_token']
         
generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
device = torch.device("cuda")
generated = generated.to(device)

model.eval();

# Top-p (nucleus) text generation (10 samples):
sample_outputs = model.generate(generated, 
                                do_sample=True,   
                                min_length=50, 
                                max_length=MAXLEN,
                                top_k=30,                                 
                                top_p=0.7,        
                                temperature=0.9,
                                repetition_penalty=2.0,
                                num_return_sequences=10
                                )

for i, sample_output in enumerate(sample_outputs):
    text = tokenizer.decode(sample_output, skip_special_tokens=True)
    a = len(title) + len(','.join(keywords))    
    print("{}: {}\n\n".format(i+1,  text[a:]))
```

1: The good news is that we’re not alone. The bad: the internet has come to its own in this week's #GoodbyeTruck memes and hashtags -- with people sharing photos from their trip on social media using them as tools for expressing themselves or making an impact online (and yes it was all thanks mostly To Kill A Mockingbird fans).
On Monday morning I posted three images taken by my husband aboard Air France flight 447 bound at LAX carrying passengers celebrating San Francisco International Airport today afternoon after disembarking over JFK airport — including one showing two young women laughing together while holding hands outside City Hall during rush hour! In fact before you start tweeting your disgust about these pictures out there if only because they were photoshopped into some sort Of Social Distortion Index, think again... pic 1 /2 We get lots more than enough hate nowadays.... but @dylanweigel isn't happy yet - he wrote "I love what little bit [sic] Instagram can do" earlier here. But then something else happened... Here are those very beautiful selfies shared early last night via MySpace/InstaTalk : You look so cute watching yourself travel across LOS ANGELES looking like YOU'RE JUST SEEN THE GIRL IN THIS VIDEO FROM STAGE HILLS *shudder* That being said though just knowing how many celebrities use hashtag tags means things aren`t exactly smooth sailing right now either … It seems Twitter will finally have someone ready who understands both humor AND sarcasm within 140 characters.. Well done everyone ;-)))))))




2: The world was shocked and saddened by the death in San Francisco on Saturday (May 12th).
A picture showing kids smoking dope with friends that has since been taken down is not only offensive to me but also sad. I’m sorry for all my fans who have gone through such an experience - from being labeled ‘supercool' as we were told it would be – unfortunately enough what happened today will send shock waves across social media everywhere because no one should ever put their life or career ahead at this point! We hope you understand now why some people took offense & thought something had just occurred pic.... — LADEE ZEN (@LadewildZen) May 11 2014




3: A picture posted on Instagram by two teenagers with the hashtag #IAMSOUTH had more than 1 million views.
The image was shared in response to an apparent train accident involving one girl and her brother who were travelling from New York City's Central Terminal (where they stayed overnight). According ot their Twitter feed it appeared that there might be trouble between them as well: "Had too many drinks & some serious alcohol poisoning."




4: Get all the very latest news in Ireland straight to your email every single day Sign up! Thank you for subscribing We have more newsletters Show me See its not dead yet


5: Our free email newsletter sends you the biggest headlines from news and sport Sign up Thank You for subscribing We have more newsletters Show me See us support your privacy notice Invalid Email
Video Loading Video Unavailable Click to play Tap again Notify centreadult who is not allowed on this site will re-enter unless it changes his or her passwords. This content would appear in future weeks Check out how we use cookies here Facebook Twitter Whatsapp Google Reddit Digg Stumbleupon Linkedin Comment




6: Get all the very latest news in Ireland straight to your email every single day Sign up! Thank you for subscribing We have more newsletters Show me See my privacy notice Invalid Email
Video Loading Video Unavailable Click here To send this article directly instead Please enter an invalid password The Independent would like nothing less than Privacy Media Facebook users will be able see exactly what is happening with their information if they wish. If someone clicks on it and then retweets or sends us something along below we'll take action - contact police at 0808 9000 3353 3857. This content was createdand maintained by two people who are themselves registered posters so that nobody else can know about them You may also get accessto these site's RSS feeds once per week Alternatively select 'edit' above which reads: "Some stuff I saw while out shopping"


7: The first time we saw the new picture was on Instagram.
“Had to call it ‘freaky funny' because I had just been drinking too much and my wife said something about how this is hilarious! It took me by surprise at what happened next," wrote one commenter in response saying "It's so cool that people are laughing with you now."




8: NEW YORK -- You might have seen one or two pictures floating around the Internet this week that were just too funny.
The photos are from L'Wren Scott's funeral in New York City and show her smiling as she leaves behind three children with Ryan Reynolds on their father James Michael Banks’ behalf at his home Sunday afternoon (June 9) inside an apartment complex outside Manhattan Beach House where he was found dead earlier Monday morning by police officers who responded to reports they believed there had been foul play following news media coverage about him hanging up while driving drunk during filming last year... but it turns out those same cops weren't looking for anything suspicious beyond what appeared like random noise coming off Chris Brown`s vehicle right next door... so instead someone decided take all these pics! And then you saw them come back online today after being taken down over Twitter? We don�t blame ya guys because now we know why things took such terrible turn Thursday night: Because people thought something wasn\'T going well between us. One picture posted early Wednesday showed blood pouring into Kevin Bacon fansite toilets; another said "I wish I could smell your urine." The first six seconds clearly indicate more drama ahead than ever before regarding alcohol abuse problems, especially since most folks didnâ€¦d want anyone else caught drinking booze prior To be clear though - none came close enough toward calling attention whatsoever.. But hey ladies & gentlemen – if any other celebs do end ups getting photoshopped together under similar circumstances — how will everyone respond?? Maybe not get very happy? Or maybe even worse-looking ones??


9: You’re in love with your picture.
Maybe it's because you don't want to be seen drunk driving on the streets all day long — or maybe just as an ordinary person would think that “cool is sexy and weird at times!'. Either way... we have some very serious appreciation for those who share their pictures online (or Instagram).




10: The world has become more and less funny with the proliferation Of These Days.
So what are we to do now? We’re not sure how many memes will surface on social media in just 24 hours but if you want your pictures taken by train or bus then it is possible that they might be posted as well so make no mistake about this: This picture was meant for teenagers! You may have seen some very inappropriate things being said towards them before – I know these photos were sent from parents who apparently don't think such behaviour goes beyond “creepy fun at school kids get drunk all day long… LOLOL'. But hopefully today's instalment doesn`t take too much getting used up over again (especially since there seems like little time left). Let us also point out here why THIS IS MY FAULT!! Here come my FOUR favourite images ever..
