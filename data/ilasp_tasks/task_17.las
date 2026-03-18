#constant(obj, o2).
#constant(obj, o7).
#constant(obj, o9).

#modeh(d0(var(obj))).
#modeh(d1(var(obj))).
#modeh(d3(var(obj))).
#modeh(d4(var(obj))).
#modeh(d7(var(obj))).
#modeh(d8(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).
#modeb(1, d5(var(obj))).
#modeb(1, d5(var(obj)), (negative)).
#modeb(1, d6(var(obj))).
#modeb(1, d6(var(obj)), (negative)).
#modeb(1, d7(var(obj))).
#modeb(1, d7(var(obj)), (negative)).
#modeb(1, d8(var(obj))).
#modeb(1, d8(var(obj)), (negative)).
#modeb(1, d9(var(obj))).
#modeb(1, d9(var(obj)), (negative)).

#pos(eg1, {
  d0(o7), d1(o9), d3(o9), d4(o9), d7(o9), d8(o7)
}, {
  d0(o2), d0(o9), d1(o2), d3(o2), d3(o7), d4(o2), d4(o7), d7(o2), d8(o2), d8(o9)
}, {
  d1(o7).
  d2(o9).
  d5(o7).
  d6(o7).
  d7(o7).
  d9(o2).
}).