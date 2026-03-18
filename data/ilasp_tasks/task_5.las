#constant(obj, o0).
#constant(obj, o1).
#constant(obj, o4).
#constant(obj, o7).
#constant(obj, o8).
#constant(obj, o9).

#modeh(d8(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
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
  d8(o4)
}, {
  d8(o0), d8(o1), d8(o7), d8(o8), d8(o9)
}, {
  d0(o7).
  d1(o1).
  d1(o4).
  d3(o4).
  d3(o9).
  d5(o0).
  d6(o4).
  d7(o0).
  d9(o8).
}).