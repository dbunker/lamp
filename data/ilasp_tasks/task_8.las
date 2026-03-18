#constant(obj, o0).
#constant(obj, o2).
#constant(obj, o5).
#constant(obj, o6).
#constant(obj, o9).

#modeh(d1(var(obj))).
#modeh(d3(var(obj))).
#modeh(d5(var(obj))).
#modeh(d6(var(obj))).

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
#modeb(1, d6(var(obj))).
#modeb(1, d6(var(obj)), (negative)).
#modeb(1, d7(var(obj))).
#modeb(1, d7(var(obj)), (negative)).
#modeb(1, d8(var(obj))).
#modeb(1, d8(var(obj)), (negative)).
#modeb(1, d9(var(obj))).
#modeb(1, d9(var(obj)), (negative)).

#pos(eg1, {
  d1(o9), d3(o9), d5(o9), d6(o9)
}, {
  d1(o0), d1(o2), d1(o5), d1(o6), d3(o0), d3(o2), d3(o5), d3(o6), d5(o0), d5(o2), d5(o5), d5(o6), d6(o0), d6(o2), d6(o5), d6(o6)
}, {
  d0(o2).
  d0(o9).
  d4(o9).
  d7(o0).
  d7(o9).
  d8(o0).
  d8(o5).
  d9(o6).
  d9(o9).
}).