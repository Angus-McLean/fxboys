/*
Shitty script to download historic forex data
Steps to Run :
- Open : http://www.histdata.com/download-free-forex-historical-data/?/metatrader/1-minute-bar-quotes
- Paste code into console
- Paste the follwing to start downloading : 

openPairs(pairLinks)

*/

pairLinks = Array.from(document.querySelectorAll('#content > div > table > tbody > tr > td > a')).map(a => a.href)

function wait (timeout) {
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve()
            }, timeout)
        })
}

async function openPairs(pairLinks) {
    debugger;
    for (let i = 0; i < pairLinks.length; i++) {
        await openPairPage(pairLinks[i])
    }
}

async function openPairPage(link) {
    return new Promise(function (res) {
        console.log('Opening Pair Link : ', link);
        wind = window.open(link)
        wind.addEventListener('load', openMonthDownloads.bind(null, wind, res), false);
    })
}

async function openMonthDownloads(wind, cb) {
    console.log('openMonthDownloads')
    monthLinks = Array.from(wind.document.querySelectorAll('#content > div > p:nth-child(6) > a')).map(a => a.href)

    for (let i = 0; i < monthLinks.length; i++) {
        console.log('Opening Pair Link : ', monthLinks[i])
        await downloadFile(wind, monthLinks[i])
        await wait(2000)
    }
    cb()
}

async function downloadFile(wind2, pageLink) {
    return new Promise(async function (res) {
        console.log('Downloading : ', pageLink)
        await wait(1000)
        monthWind = wind2.open(pageLink)
        monthWind.addEventListener('load', async function () {
            monthWind.document.getElementById('a_file').click()
            await wait(1000)
            monthWind.close()
            res()
        }, false);
    });
}