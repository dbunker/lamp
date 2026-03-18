#constant(obj, o0).
#constant(obj, o4).
#constant(obj, o7).
#constant(obj, o8).

#modeh(d6(var(obj))).

#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).
#modeb(1, d5(var(obj))).
#modeb(1, d5(var(obj)), (negative)).
#modeb(1, d6(var(obj))).
#modeb(1, d6(var(obj)), (negative)).
#modeb(1, d8(var(obj))).
#modeb(1, d8(var(obj)), (negative)).

#pos(eg1, {
  d6(o8)
}, {
  d6(o0), d6(o4), d6(o7)
}, {
  d1(o0).
  d1(o4).
  d2(o7).
  d4(o8).
  d5(o0).
  d8(o8).
}).