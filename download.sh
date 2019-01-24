
DIR=$HOME/.datasets/txt
mkdir $DIR
echo $DIR
echo '>> Download: para-nmt-50m'
wget --header="Host: doc-0k-2o-docs.googleusercontent.com" \
--header="User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.75 Safari/537.36" \
--header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8" \
--header="Accept-Language: pt-BR,pt-PT;q=0.9,pt;q=0.8,en-US;q=0.7,en;q=0.6,es-US;q=0.5,es;q=0.4,und;q=0.3" \
--header="Referer: https://drive.google.com/uc?id=1rbF3daJjCsa1-fu2GANeJd2FBXos1ugD&export=download" \
--header="Cookie: AUTH_2m333e2um3rfir4121inacj2iel2kfnm_nonce=1c2u2t642tnsa" \
--header="Connection: keep-alive" "https://doc-0k-2o-docs.googleusercontent.com/docs/securesc/8eibnt90srmbgk25k6iam51o71eao57n/5cbqlnqb8labs9s1noj0qhgo96pbb363/1548331200000/07054651981434771333/03151312306470082956/1rbF3daJjCsa1-fu2GANeJd2FBXos1ugD?e=download&nonce=1c2u2t642tnsa&user=03151312306470082956&hash=dp92k66r1tae465a0v025hd15hefgafs" \
-O $DIR/"para-nmt-50m.zip" -c
unzip $DIR/"para-nmt-50m.zip" -d $DIR

echo '>> Download: para-nmt-50m-demo'
wget --header="Host: doc-08-2o-docs.googleusercontent.com" \
--header="User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.75 Safari/537.36" \
--header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8" \
--header="Accept-Language: pt-BR,pt-PT;q=0.9,pt;q=0.8,en-US;q=0.7,en;q=0.6,es-US;q=0.5,es;q=0.4,und;q=0.3" \
--header="Referer: https://drive.google.com/uc?id=1l2liCZqWX3EfYpzv9OmVatJAEISPFihW&export=download" \
--header="Cookie: AUTH_2m333e2um3rfir4121inacj2iel2kfnm=03151312306470082956|1548331200000|re1psuc1erme1n5h0i095ro2j5t9oi24" \
--header="Connection: keep-alive" "https://doc-08-2o-docs.googleusercontent.com/docs/securesc/8eibnt90srmbgk25k6iam51o71eao57n/v3pbqjpjd0s5mfakuafprtpjil2cpbdm/1548331200000/07054651981434771333/03151312306470082956/1l2liCZqWX3EfYpzv9OmVatJAEISPFihW?e=download" \
-O $DIR/"para-nmt-50m-demo.zip" -c
unzip $DIR/"para-nmt-50m-demo.zip" -d $DIR

echo '>> Download: quora'
wget http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv -O $DIR/"quora_duplicate_questions.tsv"


echo 'DONE!!!!'