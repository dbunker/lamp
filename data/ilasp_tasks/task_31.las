#constant(obj, o0).
#constant(obj, o2).
#constant(obj, o4).
#constant(obj, o5).

#modeh(d1(var(obj))).
#modeh(d6(var(obj))).
#modeh(d7(var(obj))).
#modeh(d8(var(obj))).
#modeh(d9(var(obj))).

#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
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
  d1(o5), d6(o0), d7(o2), d7(o4), d8(o4), d9(o2), d9(o4)
}, {
  d1(o0), d1(o2), d1(o4), d6(o4), d6(o5), d7(o5), d8(o0), d8(o5), d9(o0)
}, {
  d4(o0).
  d5(o4).
  d6(o2).
  d7(o0).
  d8(o2).
  d9(o5).
}).