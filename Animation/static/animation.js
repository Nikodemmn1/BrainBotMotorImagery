const box = document.getElementById('box')
const car = document.getElementById('car')

// Od -100 do 100
const carSetPosition = (args) => {
    const {x, y} = args

    const middleX = window.innerWidth / 2
    const middleY = window.innerHeight / 2
    car.style.left = `${x / 100 * middleX * 0.6 - 50 + middleX}px`
    car.style.top = `${y / 100 * middleY * 0.6 + middleY}px`
}


const createStripe = (x, y) => {
    const stripe = document.createElement('div')
    stripe.style.background = 'white'
    stripe.style.position = 'absolute'
    stripe.style.width = '50px'
    stripe.style.height = '100px'
    stripe.style.top = y + '%'
    stripe.style.left = x + '%'
    box.appendChild(stripe)
    return stripe
}

const roadInit = () => {
    const stripeYs = [0, 20, 40, 60, 80, 100]
    stripes = stripeYs.map(y => createStripe(47, y))
}

var forward = true
var speed = 25
var stripes = []

setInterval(() => {
    stripes.map(stripe => {
        const currentTop = parseInt(stripe.style.top.slice(0, -1))
        // Tutaj tak na szybko niedokładnie policzone
        if (forward) {
            if (currentTop >= 100) stripe.style.top = -20 + '%'
            else stripe.style.top = currentTop + 1 + '%'
        } else {
            if (currentTop < -20) stripe.style.top = 100 + '%'
            else stripe.style.top = currentTop - 1 + '%'
        }
    })
}, speed);


const moveCar = (left) => {
    if (left) carSetPosition({'x': -70, 'y': 50})
    else carSetPosition({'x': 70, 'y': 50})
}


function delay(time) {
    return new Promise(resolve => setTimeout(resolve, time));
}


const readAPI = async () => {
    halo = 0
    roadInit();
    const socket = new WebSocket('ws://localhost:5000/echo');
    //await delay(2000)
    socket.addEventListener('message', data => {
        console.log("Nowa Wiadomosc")
        const commands = JSON.parse(data.data)
        console.log(commands)
        forward = commands.forward
        moveCar(commands.left)
    });
}


readAPI()
