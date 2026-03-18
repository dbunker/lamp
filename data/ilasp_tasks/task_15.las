#constant(obj, o3).
#constant(obj, o7).
#constant(obj, o8).
#constant(obj, o9).

#modeh(d8(var(obj))).

#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).
#modeb(1, d5(var(obj))).
#modeb(1, d5(var(obj)), (negative)).
#modeb(1, d7(var(obj))).
#modeb(1, d7(var(obj)), (negative)).
#modeb(1, d8(var(obj))).
#modeb(1, d8(var(obj)), (negative)).

#pos(eg1, {
  d8(o7)
}, {
  d8(o3), d8(o8), d8(o9)
}, {
  d1(o9).
  d2(o3).
  d2(o8).
  d4(o7).
  d5(o7).
  d7(o7).
}).