#constant(obj, o3).
#constant(obj, o4).
#constant(obj, o6).
#constant(obj, o7).
#constant(obj, o8).
#constant(obj, o9).

#modeh(d2(var(obj))).
#modeh(d3(var(obj))).
#modeh(d4(var(obj))).
#modeh(d5(var(obj))).
#modeh(d7(var(obj))).
#modeh(d9(var(obj))).

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
  d2(o3), d2(o4), d3(o3), d4(o3), d5(o4), d5(o7), d5(o8), d5(o9), d7(o3), d7(o4), d9(o3), d9(o4), d9(o7)
}, {
  d2(o6), d2(o7), d2(o8), d2(o9), d3(o4), d3(o6), d3(o8), d3(o9), d4(o6), d4(o7), d4(o8), d4(o9), d5(o3), d5(o6), d7(o6), d7(o7), d7(o8), d7(o9), d9(o6)
}, {
  d0(o6).
  d1(o3).
  d3(o7).
  d4(o4).
  d6(o4).
  d8(o6).
  d8(o7).
  d9(o8).
  d9(o9).
}).