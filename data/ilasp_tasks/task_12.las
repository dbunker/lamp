#constant(obj, o4).
#constant(obj, o7).
#constant(obj, o9).

#modeh(d0(var(obj))).
#modeh(d3(var(obj))).
#modeh(d4(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).
#modeb(1, d5(var(obj))).
#modeb(1, d5(var(obj)), (negative)).
#modeb(1, d8(var(obj))).
#modeb(1, d8(var(obj)), (negative)).
#modeb(1, d9(var(obj))).
#modeb(1, d9(var(obj)), (negative)).

#pos(eg1, {
  d0(o9), d3(o9), d4(o9)
}, {
  d0(o4), d0(o7), d3(o4), d3(o7), d4(o4)
}, {
  d1(o4).
  d4(o7).
  d5(o9).
  d8(o9).
  d9(o4).
  d9(o9).
}).