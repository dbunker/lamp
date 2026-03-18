#constant(obj, o4).
#constant(obj, o7).
#constant(obj, o8).
#constant(obj, o9).

#modeh(d2(var(obj))).
#modeh(d3(var(obj))).
#modeh(d5(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).
#modeb(1, d5(var(obj))).
#modeb(1, d5(var(obj)), (negative)).
#modeb(1, d7(var(obj))).
#modeb(1, d7(var(obj)), (negative)).
#modeb(1, d8(var(obj))).
#modeb(1, d8(var(obj)), (negative)).

#pos(eg1, {
  d2(o4), d3(o4), d5(o4)
}, {
  d2(o7), d2(o8), d2(o9), d3(o7), d3(o8), d3(o9), d5(o8), d5(o9)
}, {
  d0(o8).
  d4(o4).
  d5(o7).
  d7(o4).
  d8(o4).
  d8(o9).
}).